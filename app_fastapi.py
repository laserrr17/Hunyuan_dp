import random
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from hydit.config import get_args
from hydit.inference import End2End
import asyncio

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    output_path: str

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

@app.post("/generate_image")
async def generate_image(request: PromptRequest):
    try:
        enhanced_prompt = None

        # Run inference
        logger.info("Generating images with prompt: {}", request.prompt)
        height, width = args.image_size

        # Verify that prompt and necessary arguments are correctly set
        if not request.prompt:
            raise ValueError("Prompt cannot be empty.")
        
        # Generate a random seed
        random_seed = random.randint(0, 2**32 - 1)

        results = await asyncio.to_thread(gen.predict,
            request.prompt,
            height=height,
            width=width,
            seed=random_seed,  # Use random seed
            enhanced_prompt=enhanced_prompt,
            negative_prompt=args.negative,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            batch_size=args.batch_size,
            src_size_cond=args.size_cond,
            use_style_cond=args.use_style_cond,
        )

        if 'images' not in results or not results['images']:
            raise ValueError("No images generated from the model.")

        images = results['images']

        # Save images
        save_dir = Path(request.output_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = list(save_dir.glob('*.png'))
        start = max([int(f.stem) for f in all_files], default=-1) + 1

        image_paths = []
        for idx, pil_img in enumerate(images):
            save_path = save_dir / f"{idx + start}.png"
            # Use asyncio to write files concurrently
            await asyncio.to_thread(pil_img.save, save_path)
            logger.info(f"Saved to {save_path}")
            image_paths.append(str(save_path))

        return {"image_paths": image_paths}
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
# import random
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from loguru import logger
# from hydit.config import get_args
# from hydit.inference import End2End
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# app = FastAPI()
# executor = ThreadPoolExecutor(max_workers=4)

# class PromptRequest(BaseModel):
#     prompt: str
#     output_path: str

# # 全局变量，用于存储预加载的模型
# global_args = None
# global_gen = None

# async def inferencer():
#     args = get_args()
#     models_root_path = Path(args.model_root)
#     if not models_root_path.exists():
#         raise ValueError(f"`models_root` not exists: {models_root_path}")

#     # 预加载模型
#     gen = End2End(args, models_root_path)
#     return args, gen

# @app.on_event("startup")
# async def startup_event():
#     global global_args, global_gen
#     global_args, global_gen = await inferencer()
#     global_args.infer_steps = 20
#     global_args.enhance = False
#     global_args.sampler = 'dpmms'

# @app.post("/generate_image")
# async def generate_image(request: PromptRequest):
#     try:
#         enhanced_prompt = None

#         logger.info("Generating images with prompt: {}", request.prompt)
        
#         # 使用预加载的模型
#         args = global_args
#         gen = global_gen
#         height, width = args.image_size

#         if not request.prompt:
#             raise ValueError("Prompt cannot be empty.")
        
#         random_seed = random.randint(0, 2**32 - 1)

#         loop = asyncio.get_running_loop()
#         results = await loop.run_in_executor(executor, gen.predict,
#             request.prompt,
#             height,
#             width,
#             random_seed,  
#             enhanced_prompt,
#             args.negative,
#             args.infer_steps,
#             args.cfg_scale,
#             args.batch_size,
#             args.size_cond,
#             args.use_style_cond,
#         )

#         if 'images' not in results or not results['images']:
#             raise ValueError("No images generated from the model.")

#         images = results['images']

#         save_dir = Path(request.output_path)
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         all_files = list(save_dir.glob('*.png'))
#         start = max([int(f.stem) for f in all_files], default=-1) + 1

#         image_paths = []
#         for idx, pil_img in enumerate(images):
#             save_path = save_dir / f"{idx + start}.png"
#             await loop.run_in_executor(executor, pil_img.save, save_path)
#             logger.info(f"Saved to {save_path}")
#             image_paths.append(str(save_path))

#         return {"image_paths": image_paths}
#     except ValueError as ve:
#         logger.error(f"Value error: {ve}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.error(f"Error generating image: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8013)