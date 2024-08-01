import multiprocessing
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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
    # 轮询分配请求到子进程
    port = 8014 if hash(request.prompt) % 2 == 0 else 8015
    logger.info(f"Forwarding request to port {port}")
    response = requests.post(f"http://127.0.0.1:{port}/generate_image", json=request.dict())
    return response.json()

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