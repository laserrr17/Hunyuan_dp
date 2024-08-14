from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mllm.dialoggen_demo import DialogGen
import uvicorn

app = FastAPI()

# 初始化 DialogGen 模型
dialoggen_model = DialogGen(model_path='./ckpts/dialoggen')

@app.post("/dialoggen")
async def generate_response(request: dict):
    try:
        # 使用 DialogGen 生成响应
        history = request.get("history", [])
        prompt = request.get("text", "")
        skip_special = False  # 默认不跳过特殊处理

        enhanced_prompt, history = dialoggen_model(
            prompt=prompt,
            return_history=True,
            history=history,
            skip_special=skip_special
        )

        if not enhanced_prompt:
            raise HTTPException(status_code=400, detail="Failed to generate prompt.")

        # 返回字典，这样会被 FastAPI 自动转换为 JSON
        return {"result": enhanced_prompt, "history": history}
    except Exception as e:
        print("Exception occurred:", str(e))  # 输出错误信息
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)