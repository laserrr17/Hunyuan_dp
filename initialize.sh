docker run -dit     --gpus all     --init     --net=host     --uts=host     --ipc=host     --name hunyuandit_fastapi     --security-opt seccomp=unconfined     --ulimit stack=67108864     --ulimit memlock=-1     --privileged     -v $(pwd):/workspace/HunyuanDiT     -e HF_ENDPOINT=https://hf-mirror.com     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     -e CUDA_VISIBLE_DEVICES=2,3     my_hunyuandit_fastapi     python /workspace/HunyuanDiT/app_fastapi.py

docker logs -f hunyuandit_fastapi

