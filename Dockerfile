# 使用基础镜像
FROM mirrors.tencent.com/neowywang/hunyuan-dit:cuda12

# 安装必要的系统依赖，并清理缓存
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置pip镜像源为科大镜像
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple

# 安装 FastAPI 和 Uvicorn 以及其他 Python 包，并指定 diffusers 版本为 0.28.1
RUN pip install --no-cache-dir fastapi uvicorn loguru pydantic Pillow diffusers==0.28.1

# 设置工作目录
WORKDIR /workspace/HunyuanDiT