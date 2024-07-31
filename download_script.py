import os
from huggingface_hub import snapshot_download

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# 下载模型
try:
    snapshot_download(repo_id="Tencent-Hunyuan/HunyuanDiT", local_dir="./ckpts")
    print("模型下载完成")
except Exception as e:
    print(f"发生错误: {e}")