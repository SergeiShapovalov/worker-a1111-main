#!/bin/bash

echo "Worker Initiated"

# Создаем необходимые директории
mkdir -p /stable-diffusion-webui/models/Stable-diffusion
mkdir -p /stable-diffusion-webui/models/Lora
mkdir -p /stable-diffusion-webui/embeddings

# Копируем модель Flux в директорию моделей, если она еще не скопирована
if [ ! -f "/stable-diffusion-webui/models/Stable-diffusion/flux_checkpoint.safetensors" ]; then
    echo "Copying Flux model to models directory..."
    cp /model.safetensors /stable-diffusion-webui/models/Stable-diffusion/flux_checkpoint.safetensors
fi

# Загружаем EasyNegative embedding, если его еще нет
if [ ! -f "/stable-diffusion-webui/embeddings/EasyNegative.safetensors" ]; then
    echo "Downloading EasyNegative embedding..."
    wget --content-disposition -P /stable-diffusion-webui/embeddings "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors?download=true"
fi

echo "Starting Forge WebUI API"
python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt /stable-diffusion-webui/models/Stable-diffusion/flux_checkpoint.safetensors --lowram --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check --no-hashing --no-download-sd-model --forge-preset flux &

echo "Starting RunPod Handler"
python -u /rp_handler.py
