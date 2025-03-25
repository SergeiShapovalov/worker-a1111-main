import hashlib
import json
import os
import re
import subprocess
import sys
import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
from time import perf_counter
from contextlib import contextmanager
from typing import Callable, List, Dict, Any
from weights import WeightsDownloadCache
from PIL import Image
import base64
from io import BytesIO
import uuid

# Константы 
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
FLUX_CHECKPOINT_URL = "https://civitai.com/api/download/models/691639?type=Model&format=SafeTensor&size=full&fp=fp32"
sys.path.extend(["/stable-diffusion-webui"])

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

# Настройка сессии с повторными попытками
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

@contextmanager
def catchtime(tag: str) -> Callable[[], float]:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f'[Timer: {tag}]: {perf_counter() - start:.3f} seconds')

def download_base_weights(url: str, dest: str):
    """
    Загружает базовые веса модели.
    
    Args:
        url: URL для загрузки весов
        dest: Путь для сохранения весов
    """
    start = time.time()  # Засекаем время начала загрузки
    print("downloading url: ", url)
    print("downloading to: ", dest)
    
    try:
        # Используем requests вместо pget для загрузки файлов
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("downloading took: ", time.time() - start)  # Выводим время загрузки
    except Exception as e:
        print(f"Error downloading weights: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        raise e

def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)

class RunPodPredictor:
    weights_cache = WeightsDownloadCache()
    has_memory_management = False
    api = None
    
    def _download_loras(self, lora_urls: list[str]):
        lora_paths = []

        for url in lora_urls:
            if re.match(r"^https?://replicate.delivery/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/trained_model.tar", url):
                print(f"Downloading LoRA weights from - Replicate URL: {url}")
                lora_path = self.weights_cache.ensure(
                    url=url,
                    mv_from="output/flux_train_replicate/lora.safetensors",
                )
                print(f"{lora_path=}")
                lora_paths.append(lora_path)
            elif re.match(r"^https?://civitai.com/api/download/models/[0-9]+\?type=Model&format=SafeTensor", url):
                # split url to get first part of the url, everythin before '?type'
                civitai_slug = url.split('?type')[0]
                print(f"Downloading LoRA weights from - Civitai URL: {civitai_slug}")
                lora_path = self.weights_cache.ensure(url, file=True)
                lora_paths.append(lora_path)
            elif url.endswith('.safetensors'):
                print(f"Downloading LoRA weights from - safetensor URL: {url}")
                try:
                    lora_path = self.weights_cache.ensure(url, file=True)
                except Exception as e:
                    print(f"Error downloading LoRA weights: {e}")
                    continue
                print(f"{lora_path=}")
                lora_paths.append(lora_path)

        files = [os.path.join(self.weights_cache.base_dir, f) for f in os.listdir(self.weights_cache.base_dir)]
        print(f'Available loras: {files}')

        return lora_paths
    
    def setup(self, force_download_url: str = None) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Загружаем модель Flux во время сборки, чтобы ускорить генерацию
        target_dir = "/stable-diffusion-webui/models/Stable-diffusion"
        os.makedirs(target_dir, exist_ok=True)
        model_path = os.path.join(target_dir, "flux_checkpoint.safetensors")

        if not os.path.exists(model_path):
            print(f"Загружаем модель Flux...")
            download_base_weights(url=FLUX_CHECKPOINT_URL, dest=model_path)
        elif force_download_url:
            print(f"Загружаем модель Flux... {force_download_url=}")
            download_base_weights(url=force_download_url, dest=model_path)
        else:
            print(f"Модель Flux уже загружена: {model_path}, {os.path.exists(model_path)=}, {force_download_url=}")

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"

        # Безопасный импорт memory_management
        try:
            from backend import memory_management
            self.has_memory_management = True
        except ImportError as e:
            print(f"Предупреждение: Не удалось импортировать memory_management: {e}")
            self.has_memory_management = False

        # Оптимизация памяти для лучшего качества и скорости с Flux
        if self.has_memory_management:
            # Выделяем больше памяти для загрузки весов модели (90% для весов, 10% для вычислений)
            total_vram = memory_management.total_vram
            inference_memory = int(total_vram * 0.1)  # 10% для вычислений
            model_memory = total_vram - inference_memory

            memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Конвертация в байты
            print(
                f"[GPU Setting] Выделено {model_memory} MB для весов модели и {inference_memory} MB для вычислений"
            )

            # Настройка Swap Method на ASYNC для лучшей производительности
            try:
                from backend import stream
                # Для Flux рекомендуется ASYNC метод, который может быть до 30% быстрее
                stream.stream_activated = True  # True = ASYNC, False = Queue
                print("[GPU Setting] Установлен ASYNC метод загрузки для лучшей производительности")

                # Настройка Swap Location на Shared для лучшей производительности
                memory_management.PIN_SHARED_MEMORY = True  # True = Shared, False = CPU
                print("[GPU Setting] Установлен Shared метод хранения для лучшей производительности")
            except ImportError as e:
                print(f"Предупреждение: Не удалось импортировать stream: {e}")
        else:
            print("[GPU Setting] memory_management не доступен, используются настройки по умолчанию")

    def predict(self, input_data):
        """Run inference on the input data"""
        try:
            # Извлекаем параметры из запроса
            prompt = input_data.get("prompt", "")
            width = input_data.get("width", 0)
            height = input_data.get("height", 0)
            num_outputs = input_data.get("num_outputs", 1)
            sampler = input_data.get("sampler", "Euler")
            scheduler = input_data.get("scheduler", "Simple")
            num_inference_steps = input_data.get("num_inference_steps", 8)
            guidance_scale = input_data.get("guidance_scale", 1.0)
            distilled_guidance_scale = input_data.get("distilled_guidance_scale", 3.5)
            seed = input_data.get("seed", -1)
            enable_hr = input_data.get("enable_hr", True)
            hr_upscaler = input_data.get("hr_upscaler", "R-ESRGAN 4x+")
            hr_steps = input_data.get("hr_steps", 8)
            hr_scale = input_data.get("hr_scale", 1.3)
            denoising_strength = input_data.get("denoising_strength", 0.3)
            debug_flux_checkpoint_url = input_data.get("debug_flux_checkpoint_url", "")
            enable_clip_l = input_data.get("enable_clip_l", False)
            enable_t5xxl_fp16 = input_data.get("enable_t5xxl_fp16", False)
            enable_ae = input_data.get("enable_ae", False)
            force_model_reload = input_data.get("force_model_reload", False)
            forge_unet_storage_dtype = input_data.get("forge_unet_storage_dtype", "bnb-nf4 (fp16 LoRA)")
            image = input_data.get("image", None)
            prompt_strength = input_data.get("prompt_strength", 0.8)
            aspect_ratio = input_data.get("aspect_ratio", "9:16")
            output_format = input_data.get("output_format", "webp")
            lora_urls = input_data.get("lora_urls", [])
            lora_scales = input_data.get("lora_scales", [1.0] * len(lora_urls))
            
            # Импортируем необходимые модули
            from modules import shared
            
            # Устанавливаем forge_preset на 'flux'
            shared.opts.set('forge_preset', 'flux')
            
            # Устанавливаем чекпоинт
            shared.opts.set('sd_model_checkpoint', 'flux_checkpoint.safetensors')
            
            # Импортируем модули для работы с API
            from modules.extra_networks import ExtraNetworkParams
            from modules import scripts
            from modules.api.models import (
                StableDiffusionTxt2ImgProcessingAPI,
                StableDiffusionImg2ImgProcessingAPI
            )
            from modules_forge.main_entry import forge_unet_storage_dtype_options
            from backend.args import dynamic_args
            
            # Получаем параметры для forge_unet_storage_dtype
            forge_unet_storage_dtype_value, online_lora = forge_unet_storage_dtype_options.get(
                forge_unet_storage_dtype, (None, False),
            )
            
            print(f"Setting {forge_unet_storage_dtype_value=}, {online_lora=}")
            shared.opts.set('forge_unet_storage_dtype', forge_unet_storage_dtype_value)
            dynamic_args['online_lora'] = online_lora
            
            # Проверяем, нужно ли загрузить модель заново
            if debug_flux_checkpoint_url:
                self.setup(force_download_url=debug_flux_checkpoint_url)
            
            # Загружаем LoRA модели
            lora_paths = self._download_loras(lora_urls)
            
            # Устанавливаем размеры изображения на основе соотношения сторон, если не указаны
            if (not width) or (not height):
                width, height = ASPECT_RATIOS[aspect_ratio]
            
            # Формируем payload для API
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "batch_size": num_outputs,
                "steps": num_inference_steps,
                "cfg_scale": guidance_scale,
                "seed": seed,
                "do_not_save_samples": True,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "enable_hr": enable_hr,
                "hr_upscaler": hr_upscaler,
                "hr_second_pass_steps": hr_steps,
                "denoising_strength": denoising_strength if enable_hr else None,
                "hr_scale": hr_scale,
                "distilled_cfg_scale": distilled_guidance_scale,
                "hr_additional_modules": [],
            }
            
            # Добавляем параметры для img2img, если указано изображение
            if image:
                payload['denoising_strength'] = prompt_strength
                payload['init_images'] = [image]
                payload['resize_mode'] = 1
            
            alwayson_scripts = {}
            
            # Добавляем все скрипты в payload, если они есть
            if alwayson_scripts:
                payload["alwayson_scripts"] = alwayson_scripts
            
            print(f"Финальный пейлоад: {payload=}")
            
            # Формируем дополнительные параметры запроса
            req = dict(
                forge_unet_storage_dtype=forge_unet_storage_dtype_value,
                force_model_reload=force_model_reload,
                extra_network_data={
                    "lora": [
                        ExtraNetworkParams(
                            items=[
                                lora_path.split('/')[-1].split('.safetensors')[0],
                                str(lora_scale)
                            ]
                        )
                        for lora_path, lora_scale in zip(lora_paths, lora_scales)
                    ]
                },
                additional_modules={
                    "clip_l.safetensors": enable_clip_l,
                    "t5xxl_fp16.safetensors": enable_t5xxl_fp16,
                    "ae.safetensors": enable_ae,
                },
            )
            
            for lora in req['extra_network_data']['lora']:
                print(f"LoRA: {lora.items=}")
            
            # Выполняем запрос к API
            with catchtime(tag="Total Prediction Time"):
                if image:
                    # Для img2img
                    response = automatic_session.post(
                        url=f'{LOCAL_URL}/img2img',
                        json={
                            **req,
                            "img2imgreq": payload
                        },
                        timeout=600
                    )
                else:
                    # Для txt2img
                    response = automatic_session.post(
                        url=f'{LOCAL_URL}/txt2img',
                        json={
                            **req,
                            "txt2imgreq": payload
                        },
                        timeout=600
                    )
            
            # Обрабатываем ответ
            response_data = response.json()
            
            # Преобразуем изображения в нужный формат
            images = []
            if 'images' in response_data:
                info = json.loads(response_data.get('info', '{}'))
                all_seeds = info.get('all_seeds', [-1] * len(response_data['images']))
                
                with catchtime(tag="Total Encode Time"):
                    for i, img_data in enumerate(response_data['images']):
                        seed = all_seeds[i] if i < len(all_seeds) else -1
                        
                        # Декодируем изображение из base64
                        img_bytes = BytesIO(base64.b64decode(img_data))
                        img = Image.open(img_bytes)
                        
                        # Сохраняем изображение во временный файл
                        temp_filename = f"/tmp/{seed}-{uuid.uuid4()}.{output_format}"
                        
                        if output_format != 'png':
                            img.save(temp_filename, format=output_format.upper(), quality=100, optimize=True)
                        else:
                            img.save(temp_filename, format=output_format.upper())
                        
                        # Читаем файл как base64 для возврата
                        with open(temp_filename, "rb") as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Добавляем информацию об изображении
                        images.append({
                            "image": img_base64,
                            "seed": seed,
                            "width": img.width,
                            "height": img.height,
                            "format": output_format
                        })
                        
                        # Удаляем временный файл
                        os.remove(temp_filename)
            
            return {
                "images": images,
                "parameters": payload
            }
        
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

# Создаем экземпляр предиктора
predictor = RunPodPredictor()

# RunPod Handler
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    try:
        # Получаем входные данные
        input_data = event.get("input", {})
        
        # Запускаем инференс
        result = predictor.predict(input_data)
        
        return result
    except Exception as e:
        print(f"Error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Инициализируем предиктор
    predictor.setup()
    
    # Ждем, пока API будет готов
    wait_for_service(url=f'{LOCAL_URL}/txt2img')
    
    print("WebUI API Service is ready. Starting RunPod...")
    
    # Запускаем RunPod сервер
    runpod.serverless.start({"handler": handler})
