import torch
from diffusers import HunyuanDiTPipeline
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载模型
pipe = HunyuanDiTPipeline.from_pretrained(
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    device_map=None
)
pipe.to("cuda")

# 使用的提示词
prompt = "一个宇航员在骑马"

# 保存图像的目录
save_dir = Path("results")
save_dir.mkdir(parents=True, exist_ok=True)

# 获取目录中现有的最大文件索引
all_files = list(save_dir.glob('*.png'))
start = max([int(f.stem) for f in all_files], default=-1) + 1

def generate_image(pipe, prompt, index):
    try:
        image = pipe(prompt, num_inference_steps=25).images[0]
        return image
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

async def save_image_async(image, index):
    save_path = save_dir / f"{index}.png"
    await asyncio.to_thread(image.save, save_path)
    logger.info(f"Saved to {save_path}")
    return str(save_path)

async def main():
    num_images = 10
    image_paths = []

    # 使用线程池同时生成10张图像
    with ThreadPoolExecutor(max_workers=num_images) as executor:
        futures = [executor.submit(generate_image, pipe, prompt, i + start) for i in range(num_images)]
    
    # 获取生成的图像并异步保存
    images = [future.result() for future in futures if future.result() is not None]
    save_tasks = [save_image_async(image, i + start) for i, image in enumerate(images)]
    image_paths = await asyncio.gather(*save_tasks)

    logger.info(f"Generated and saved images: {image_paths}")

if __name__ == "__main__":
    asyncio.run(main())