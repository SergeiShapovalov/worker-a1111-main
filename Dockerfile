# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.36.2 as download

COPY builder/clone.sh /clone.sh

# Clone the repos and clean unnecessary files
RUN . /clone.sh taming-transformers https://github.com/CompVis/taming-transformers.git 24268930bf1dce879235a7fddd0b2355b84d7ea6 && \
    rm -rf data assets **/*.ipynb

RUN . /clone.sh stable-diffusion-stability-ai https://github.com/Stability-AI/stablediffusion.git 47b6b607fdd31875c9279cd2f4f16b92e4ea958e && \
    rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh CodeFormer https://github.com/sczhou/CodeFormer.git c5b4593074ba6214284d6acd5f1719b6c5d739af && \
    rm -rf assets inputs

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 48211a1594f1321b00f14c9f7a5b4813144b2fb9 && \
    . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git 5b3af030dd83e0297272d861c19477735d0317ec && \
    . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator 2486589f24165c8e3b303f84e9dbbea318df83e8 && \
    . /clone.sh generative-models https://github.com/Stability-AI/generative-models 45c443b316737a4ab6e40413d7794a7f5657c19f
# Загрузка модели Flux будет выполнена в финальном образе, а не в стадии download



# ---------------------------------------------------------------------------- #
#                        Stage 3: Build the final image                        #
# ---------------------------------------------------------------------------- #
FROM python:3.11-slim as build_final_image

ARG SHA=5ef669de080814067961f28357256e8fe27544f4

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    LD_PRELOAD=libtcmalloc.so \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN export COMMANDLINE_ARGS="--skip-torch-cuda-test --precision full --no-half"
RUN export TORCH_COMMAND='pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu124'

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/cache --mount=type=cache,target=/root/.cache/pip \
    pip install https://download.pytorch.org/whl/nightly/cu124/torch-2.6.0.dev20240918%2Bcu124-cp311-cp311-linux_x86_64.whl \
    https://download.pytorch.org/whl/nightly/cu124/torchaudio-2.5.0.dev20240918%2Bcu124-cp311-cp311-linux_x86_64.whl \
    https://download.pytorch.org/whl/nightly/cu124/torchvision-0.20.0.dev20240918%2Bcu124-cp311-cp311-linux_x86_64.whl \
    https://download.pytorch.org/whl/nightly/pytorch_triton-3.1.0%2B5fe38ffd73-cp311-cp311-linux_x86_64.whl

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/SergeiShapovalov/ForgeRunPod.git stable-diffusion-webui && \
    cd stable-diffusion-webui
#&& \ pip install -r requirements_versions.txt

COPY --from=download /repositories/ ${ROOT}/repositories/
RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate

# Загрузка модели Flux
RUN mkdir -p ${ROOT}/models/Stable-diffusion && \
    wget -q -O ${ROOT}/models/Stable-diffusion/flux_checkpoint.safetensors \
    "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=full&fp=fp32&token=18b51174c4d9ae0451a3dedce1946ce3"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pip==23.3.1 && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Копируем файлы из директории src
COPY src/rp_handler.py /rp_handler.py
COPY src/start.sh /start.sh
COPY src/weights.py /weights.py

COPY builder/cache.py /stable-diffusion-webui/cache.py
# Пропускаем запуск cache.py при сборке, он будет запущен при старте контейнера на RunPod, где есть GPU
# RUN cd /stable-diffusion-webui && python cache.py --use-cpu=all --ckpt /stable-diffusion-webui/models/Stable-diffusion/flux_checkpoint.safetensors

# Cleanup section (Worker Template)
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Set permissions and specify the command to run
RUN chmod +x /start.sh
CMD /start.sh
