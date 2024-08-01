import multiprocessing
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import uvicorn
import random
import asyncio
import aiohttp

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    output_path: str

# 启动子进程
def start_child_process(port, cuda_device):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    uvicorn.run("child:app", host="0.0.0.0", port=port, reload=False)

@app.post("/generate_image")
async def generate_image(request: PromptRequest):
    # 随机选择一个子进程的端口
    port = random.choice([8014, 8015])
    logger.info(f"Forwarding request to port {port}")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://127.0.0.1:{port}/generate_image", json=request.dict()) as response:
            return await response.json()

if __name__ == "__main__":
    ports_devices = [(8014, "2"), (8015, "3")]
    processes = []

    for port, device in ports_devices:
        p = multiprocessing.Process(target=start_child_process, args=(port, device))
        p.start()
        processes.append(p)

    uvicorn.run(app, host="0.0.0.0", port=8013)

    for p in processes:
        p.join()