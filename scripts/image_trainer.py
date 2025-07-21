#!/usr/bin/env python3
"""
Optimized image model training (SDXL or Flux) for G.O.D. Tournament
"""

import argparse
import asyncio
import os
import subprocess
import sys

import toml

# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path


def get_optimized_config_for_model_type(model_type, dataset_size=None):
    """Get optimized parameters based on model type and dataset"""
    
    if model_type == ImageModelType.SDXL.value:
        return {
            # Оптимизированные параметры для SDXL
            "learning_rate": 1e-4,  # Выше стандартного для быстрой сходимости
            "unet_lr": 1e-4,
            "text_encoder_lr": 5e-5,  # Выше для лучшего понимания промптов
            "optimizer_type": "AdamW",  # Полный AdamW вместо 8bit для качества
            "optimizer_args": ["betas=(0.9,0.999)", "weight_decay=0.01"],
            "network_dim": 128,  # Увеличен rank для лучшего качества
            "network_alpha": 64,  # Половина от dim для стабильности
            "train_batch_size": 6,  # Увеличено для H200
            "gradient_accumulation_steps": 2,
            "max_train_steps": 2000,  # Больше шагов для качества
            "lr_scheduler": "cosine_with_restarts",
            "lr_scheduler_num_cycles": 3,  # Перезапуски для выхода из локальных минимумов
            "lr_warmup_steps": 100,
            "min_snr_gamma": 5,
            "noise_offset": 0.1,  # Для лучшей генерации темных/светлых изображений
            "adaptive_noise_scale": 0.0,  # Отключено для стабильности
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "xformers": True,
            "max_data_loader_n_workers": 8,  # Больше воркеров для H200
            "persistent_data_loader_workers": True,
            "mixed_precision": "bf16",
            "full_bf16": True,
            "save_precision": "bf16",
            "save_model_as": "safetensors",
            "save_every_n_epochs": 10,
            "sample_every_n_epochs": 5,
            "sample_prompts": "",  # Будет заполнено из датасета
            "wandb_api_key": "",  # Отключаем для турнира
            "lowram": False,  # У нас достаточно памяти
            "max_token_length": 225,  # Увеличено для сложных промптов
            "clip_skip": 2,  # Стандарт для аниме/стилизованных изображений
            "prior_loss_weight": 0.5,  # Баланс между сохранением стиля и обучением
        }
    elif model_type == ImageModelType.FLUX.value:
        return {
            # Оптимизированные параметры для Flux
            "learning_rate": 4e-4,  # Выше для Flux
            "unet_lr": 4e-4,
            "text_encoder_lr": [2e-4, 2e-4],  # Для dual text encoders
            "optimizer_type": "Adafactor",  # Лучше для Flux
            "optimizer_args": ["scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01"],
            "network_dim": 256,  # Больше для Flux
            "network_alpha": 128,
            "network_args": ["train_norm=True", "train_conv=True"],  # Дополнительные слои
            "train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "epoch": 150,  # Больше эпох для Flux
            "max_train_steps": 3000,
            "lr_scheduler": "constant_with_warmup",
            "lr_warmup_steps": 200,
            "timestep_sampling": "sigmoid",  # Лучше для качества
            "discrete_flow_shift": 3.1582,  # Оптимально для Flux
            "model_prediction_type": "raw",
            "guidance_scale": 3.5,  # Выше для лучшего следования промптам
            "loss_type": "l2",
            "huber_schedule": "exponential",
            "huber_c": 0.1,
            "cache_text_encoder_outputs": True,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "cpu_offload_checkpointing": True,  # Для экономии VRAM
            "mixed_precision": "bf16",
            "full_bf16": True,
            "save_precision": "bf16",
            "max_data_loader_n_workers": 8,
            "persistent_data_loader_workers": True,
            "highvram": False,  # Баланс между скоростью и стабильностью
            "sample_every_n_epochs": 25,
            "sample_prompts": "",
            "wandb_api_key": "",
            "apply_t5_attn_mask": True,
            "split_mode": True,  # Split computation для больших батчей
            "train_t5": True,  # Обучаем T5 для лучшего понимания
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_config(task_id, model, model_type, expected_repo_name):
    """Create the optimized diffusion config file"""
    
    # Load base config
    if os.path.exists("/workspace/core/config"):
        config_path = "/workspace/core/config"
        sdxl_path = f"{config_path}/base_diffusion_sdxl.toml"
        flux_path = f"{config_path}/base_diffusion_flux.toml"
    else:
        sdxl_path = cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL
        flux_path = cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX

    # Load appropriate config template
    if model_type == ImageModelType.SDXL.value:
        with open(sdxl_path, "r") as file:
            config = toml.load(file)
    elif model_type == ImageModelType.FLUX.value:
        with open(flux_path, "r") as file:
            config = toml.load(file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Apply optimized parameters
    optimized_params = get_optimized_config_for_model_type(model_type)
    config.update(optimized_params)

    # Update paths
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    output_dir = f"{train_cst.IMAGE_CONTAINER_SAVE_PATH}{task_id}/{expected_repo_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Model-specific adjustments based on dataset size
    dataset_path = config["train_data_dir"]
    if os.path.exists(dataset_path):
        num_images = len([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if num_images < 20:
            # Small dataset optimizations
            config["train_batch_size"] = max(1, config["train_batch_size"] // 2)
            config["max_train_steps"] = min(1500, config.get("max_train_steps", 1500))
            config["network_dim"] = min(64, config["network_dim"])
        elif num_images > 100:
            # Large dataset optimizations
            config["train_batch_size"] = min(8, config["train_batch_size"] * 2)
            config["gradient_accumulation_steps"] = 1

    # Save config to file
    config_path = os.path.join("/dataset/configs", f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created optimized config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting optimized {model_type} training with config: {config_path}", flush=True)

    # Environment optimizations
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    env["OMP_NUM_THREADS"] = "8"  # Optimize CPU threads

    training_command = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "8",
        "--dynamo_backend", "no",  # Отключаем для стабильности с diffusion
        f"/app/sd-scripts/{model_type}_train_network.py",
        "--config_file", config_path
    ]

    try:
        print("Starting optimized training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING OPTIMIZED IMAGE TRAINING SCRIPT---", flush=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimized Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    args = parser.parse_args()

    # Create required directories
    os.makedirs("/dataset/configs", exist_ok=True)
    os.makedirs("/dataset/outputs", exist_ok=True)
    os.makedirs("/dataset/images", exist_ok=True)

    model_folder = args.model.replace("/", "--")
    model_path = get_model_path(f"{train_cst.CACHE_PATH}/models/{model_folder}")

    # Create optimized config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
    )

    # Prepare dataset with optimized settings
    print("Preparing dataset with optimizations...", flush=True)

    # Set DIFFUSION_DATASET_DIR to environment variable if available
    original_dataset_dir = cst.DIFFUSION_DATASET_DIR
    if os.environ.get("DATASET_DIR"):
        cst.DIFFUSION_DATASET_DIR = os.environ.get("DATASET_DIR")

    # Optimize repeats based on dataset size and model type
    if args.model_type == ImageModelType.SDXL.value:
        repeats = 15  # Увеличено для лучшего обучения
    else:  # Flux
        repeats = 2  # Меньше для Flux из-за его эффективности

    prepare_dataset(
        training_images_zip_path=f"{train_cst.CACHE_PATH}/datasets/{args.task_id}.zip",
        training_images_repeat=repeats,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
    )

    # Restore original value
    cst.DIFFUSION_DATASET_DIR = original_dataset_dir

    # Run optimized training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
