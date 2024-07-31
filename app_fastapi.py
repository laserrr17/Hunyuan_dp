# import random
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from loguru import logger
# from hydit.config import get_args
# from hydit.inference import End2End
# import asyncio
# from asyncio import Lock

# app = FastAPI()

# class PromptRequest(BaseModel):
#     prompt: str
#     output_path: str

# # 初始化共享计数器和锁
# concurrent_requests = 0
# concurrent_requests_lock = Lock()
# predict_lock = Lock()

# async def inferencer():
#     args = get_args()
#     models_root_path = Path(args.model_root)
#     if not models_root_path.exists():
#         raise ValueError(f"`models_root` not exists: {models_root_path}")

#     # Load models
#     gen = End2End(args, models_root_path)
#     return args, gen

# @app.on_event("startup")
# async def startup_event():
#     global args, gen
#     args, gen = await inferencer()
#     args.infer_steps = 20
#     args.enhance = False
#     args.sampler = 'dpmms'

# @app.post("/generate_image")
# async def generate_image(request: PromptRequest):
#     global concurrent_requests
#     async with concurrent_requests_lock:
#         concurrent_requests += 1

#     try:
#         enhanced_prompt = None

#         # Run inference
#         logger.info("Generating images with prompt: {}", request.prompt)
#         height, width = args.image_size

#         # Verify that prompt and necessary arguments are correctly set
#         if not request.prompt:
#             raise ValueError("Prompt cannot be empty.")
        
#         # Generate a random seed
#         random_seed = random.randint(0, 2**32 - 1)

#         # 动态调整batch_size，根据当前并发请求数量
#         dynamic_batch_size = max(1, args.batch_size - concurrent_requests)

#         async with predict_lock:
#             results = await asyncio.to_thread(gen.predict,
#                 request.prompt,
#                 height=height,
#                 width=width,
#                 seed=random_seed,  # Use random seed
#                 enhanced_prompt=enhanced_prompt,
#                 negative_prompt=args.negative,
#                 infer_steps=args.infer_steps,
#                 guidance_scale=args.cfg_scale,
#                 batch_size=dynamic_batch_size,  # 使用动态的batch_size
#                 src_size_cond=args.size_cond,
#                 use_style_cond=args.use_style_cond,
#             )

#         if 'images' not in results or not results['images']:
#             raise ValueError("No images generated from the model.")

#         images = results['images']

#         # Save images
#         save_dir = Path(request.output_path)
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         all_files = list(save_dir.glob('*.png'))
#         start = max([int(f.stem) for f in all_files], default=-1) + 1

#         image_paths = []
#         for idx, pil_img in enumerate(images):
#             save_path = save_dir / f"{idx + start}.png"
#             # Use asyncio to write files concurrently
#             await asyncio.to_thread(pil_img.save, save_path)
#             logger.info(f"Saved to {save_path}")
#             image_paths.append(str(save_path))

#         return {"image_paths": image_paths}
#     except ValueError as ve:
#         logger.error(f"Value error: {ve}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.error(f"Error generating image: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")
#     finally:
#         async with concurrent_requests_lock:
#             concurrent_requests -= 1

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8013)
import random
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from hydit.config import get_args
from hydit.inference import End2End
import asyncio
from asyncio import Semaphore, Queue

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    output_path: str

# 初始化共享计数器和信号量
batch_size = 3
semaphore = Semaphore(batch_size)
request_queue = Queue()

async def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)
    return args, gen

@app.on_event("startup")
async def startup_event():
    global args, gen
    args, gen = await inferencer()
    args.infer_steps = 20
    args.enhance = False
    args.sampler = 'dpmms'
    
    # 启动批处理任务
    asyncio.create_task(process_batch())

async def process_batch():
    global request_queue
    while True:
        batch = []
        # 获取一批请求
        for _ in range(batch_size):
            batch.append(await request_queue.get())

        try:
            for req in batch:
                request = req['request']
                response_future = req['response']

                # 运行推理
                logger.info("Generating image with prompt: {}", request.prompt)
                height, width = args.image_size

                # 生成随机种子
                random_seed = random.randint(0, 2**32 - 1)

                results = await asyncio.to_thread(gen.predict,
                    request.prompt,
                    height=height,
                    width=width,
                    seed=random_seed,
                    enhanced_prompt=None,
                    negative_prompt=args.negative,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    batch_size=1,  # 单独处理每个请求
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
            for req in batch:
                req['response'].set_exception(HTTPException(status_code=400, detail=str(ve)))
        except Exception as e:
            for req in batch:
                req['response'].set_exception(HTTPException(status_code=500, detail="Internal server error"))

@app.post("/generate_image")
async def generate_image(request: PromptRequest):
    response_future = asyncio.get_event_loop().create_future()
    await request_queue.put({'request': request, 'response': response_future})

    async with semaphore:
        return await response_future

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)