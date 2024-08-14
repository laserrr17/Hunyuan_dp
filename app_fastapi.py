from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
from hydit.config import get_args
from hydit.inference import End2End
import asyncio
from asyncio import Semaphore, Queue
from pathlib import Path
import random
from mllm.dialoggen_demo import DialogGen

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    output_path: str

class ImageAdjustRequest(BaseModel):
    session_id: str
    prompt: str

# Initialize semaphore and queue
semaphore = Semaphore(3)
request_queue = Queue()
active_sessions = {}

async def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models including the enhancer
    gen = End2End(args, models_root_path)
    return args, gen

@app.on_event("startup")
async def startup_event():
    global args, gen, dialoggen_model
    dialoggen_model = DialogGen(model_path='./ckpts/dialoggen', args.load_4bit)
    args, gen = await inferencer()
    args.infer_steps = 20
    args.enhance = False
    args.sampler = 'dpmms'
    
    # Start the process queue task
    asyncio.create_task(process_queue())

async def process_queue():
    while True:
        req = await request_queue.get()
        try:
            request = req['request']
            response_future = req['response']

            # Run inference
            logger.info("Generating image with prompt: {}", request.prompt)
            height, width = args.image_size

            # Generate random seed
            random_seed = 42

            results = await asyncio.to_thread(gen.predict,
                request.prompt,
                height=height,
                width=width,
                seed=random_seed,
                enhanced_prompt=None,
                negative_prompt=args.negative,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                batch_size=1,
                src_size_cond=args.size_cond,
                use_style_cond=args.use_style_cond,
            )

            if 'images' not in results or not results['images']:
                raise ValueError("No images generated from the model.")

            images = results['images']

            save_dir = Path(request.output_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            all_files = list(save_dir.glob('*.png'))
            start = max([int(f.stem) for f in all_files], default=-1) + 1

            image_paths = []
            for idx, pil_img in enumerate(images):
                save_path = save_dir / f"{idx + start}.png"
                await asyncio.to_thread(pil_img.save, save_path)
                logger.info(f"Saved to {save_path}")
                image_paths.append(str(save_path))

            response_future.set_result({"image_paths": image_paths})

        except ValueError as ve:
            req['response'].set_exception(HTTPException(status_code=400, detail=str(ve)))
        except Exception as e:
            req['response'].set_exception(HTTPException(status_code=500, detail="Internal server error"))

@app.post("/generate_image")
async def generate_image(request: PromptRequest):
    response_future = asyncio.get_event_loop().create_future()
    session_id = str(random.randint(1000, 9999))  # Simple session ID generation
    active_sessions[session_id] = {"prompt": request.prompt, "output_path": request.output_path}
    await request_queue.put({'request': request, 'response': response_future})
    result = await response_future
    result['session_id'] = session_id
    return result

@app.post("/adjust_image")
async def adjust_image(request: ImageAdjustRequest):
    session_id = request.session_id
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = active_sessions[session_id]
    session_prompt = session_data['prompt']
    session_output_path = session_data['output_path']
    session_history = session_data.get('history', [])

    # 确保 session_history 格式正确
    if not isinstance(session_history, list):
        session_history = []

    # 调用 dialoggen_model 直接生成增强的 prompt
    enhanced_prompt, updated_history = dialoggen_model(prompt=request.prompt + active_sessions[session_id]['prompt'], return_history=True, history=session_history)

    # 更新 prompt 和 history
    active_sessions[session_id]['prompt'] = enhanced_prompt
    active_sessions[session_id]['history'] = updated_history

    new_request = PromptRequest(prompt=enhanced_prompt, output_path=session_output_path)

    # 使用 asyncio.get_running_loop() 以避免潜在问题
    response_future = asyncio.get_running_loop().create_future()
    await request_queue.put({'request': new_request, 'response': response_future})
    return await response_future

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)