#!/bin/bash

echo "Worker Initiated"

# Создаем необходимые директории
mkdir -p /stable-diffusion-webui/models/Stable-diffusion
mkdir -p /stable-diffusion-webui/models/Lora
mkdir -p /stable-diffusion-webui/embeddings

# Проверяем наличие модели Flux и загружаем ее при необходимости
FLUX_MODEL_PATH="/stable-diffusion-webui/models/Stable-diffusion/flux_checkpoint.safetensors"
if [ ! -f "$FLUX_MODEL_PATH" ]; then
    echo "Downloading Flux model..."
    wget -q -O "$FLUX_MODEL_PATH" "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=full&fp=fp32&token=18b51174c4d9ae0451a3dedce1946ce3"
    echo "Flux model downloaded successfully"
else
    echo "Using existing Flux model from $FLUX_MODEL_PATH"
fi

# Загружаем EasyNegative embedding, если его еще нет
if [ ! -f "/stable-diffusion-webui/embeddings/EasyNegative.safetensors" ]; then
    echo "Downloading EasyNegative embedding..."
    wget --content-disposition -P /stable-diffusion-webui/embeddings "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors?download=true"
fi
# Запускаем cache.py для предварительной загрузки модели в GPU
echo "Preloading model with cache.py..."
cd /stable-diffusion-webui && python cache.py --ckpt /stable-diffusion-webui/models/Stable-diffusion/flux_checkpoint.safetensors

echo "Starting Forge WebUI API"

python /stable-diffusion-webui/webui.py --skip-python-version-check --skip-torch-cuda-test --skip-install --ckpt flux_checkpoint.safetensors --lowram --disable-safe-unpickle --port 3000 --api --nowebui --skip-version-check --no-hashing --no-download-sd-model --forge-preset flux &

echo "Starting RunPod Handler"
python -u /rp_handler.py
