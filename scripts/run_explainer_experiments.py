#!/usr/bin/env python3
"""
Script to run experiments with different explainer models and non-activating sources
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

# Configuration
BASE_MODEL = "EleutherAI/pythia-160m"
SPARSE_MODEL = "EleutherAI/Pythia-160m-SST-k32-32k"
SPARSE_MODEL_NAME = "pythiaST"  # Short name for directory structure
HOOKPOINT = "layers.3.mlp"
DATASET_REPO = "EleutherAI/rpj-v2-sample"
DATASET_NAME = "default"
DATASET_COLUMN = "raw_content"
MAX_LATENTS = 100  # Main configuration parameter
DIR_NAME = "Contrastive Experiments"
THINKING_MODE = False  # Set to True to enable thinking mode
USE_SEPARATE_SCORER = False

# Non-activating source configurations to test
NON_ACTIVATING_SOURCES = [
    "random", 
    # "co-occurrence",
    # "decoder_similarity",
    # "faiss",
]

# Contrastive experiment configurations
CONTRASTIVE_CONFIGS = [
    # Format: (use_contrastive_explainer, use_contrastive_scorer, description)
    (False, False, "baseline"),  # Standard behavior
    # (True, False, "contrastive_explainer_only"),  # Use contrastive explainer only
    # (False, True, "contrastive_scorer_only"),  # Use contrastive scorer only  
    # (True, True, "both_contrastive"),  # Use both contrastive explainer and scorer
]

# Train type configurations to test
TRAIN_TYPES = [
    "quantiles",
    "top", 
    # "random",
    # "mix",  # Uncomment if needed
]

# Explainer models to test
EXPLAINER_MODELS = [
    # "RedHatAI/gemma-3-4b-it-quantized.w4a16",
    # "RedHatAI/Qwen3-4B-quantized.w4a16",
    # "RedHatAI/gemma-3-12b-it-quantized.w4a16",
    # "RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "RedHatAI/Qwen3-14B-quantized.w4a16",
    # "RedHatAI/Qwen3-32B-quantized.w4a16",
    # "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    # "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    # "Transluce/llama_8b_explainer"
    # "openai/gpt-oss-20b",
]

def get_model_name(model_path: str) -> str:
    """Extract a clean model name from the full model path."""
    return model_path.split('/')[-1].replace('-', '_').replace('.', '_')

def get_base_dir() -> Path:
    """Get the base directory for this sparse model."""
    return Path("results") / SPARSE_MODEL_NAME

def get_cache_dir() -> Path:
    """Get the shared cache directory."""
    return get_base_dir() / "cache"

def get_experiment_dir(explainer_model: str, non_activating_source: str, 
                      contrastive_config: Tuple[bool, bool, str], train_type: str) -> Path:
    """Get the experiment directory for a specific configuration."""
    model_name = get_model_name(explainer_model)
    use_contrastive_explainer, use_contrastive_scorer, contrastive_desc = contrastive_config
    
    # Build experiment name components
    components = [
        SPARSE_MODEL_NAME, 
        model_name,
        non_activating_source,
        contrastive_desc,
        train_type
    ]
    
    if THINKING_MODE:
        components.append("thinking")
    
    experiment_name = "_".join(components)
    return get_base_dir() / DIR_NAME / experiment_name

def setup_shared_cache() -> None:
    """Set up shared activation cache."""
    print("Setting up shared activation cache...")
    
    cache_path = get_cache_dir()
    
    if cache_path.exists():
        print(f"Shared cache already exists at {cache_path}")
        # Check the layer structure
        layer_dirs = list(cache_path.glob("latents/layers.*"))
        if layer_dirs:
            print(f"Cache contains {len(layer_dirs)} layer directories: {[d.name for d in layer_dirs]}")
        else:
            print("Cache directory exists but is empty - will be populated during first run")
    else:
        print(f"Cache directory does not exist at {cache_path} - will be created during first run")
    print()

def run_experiment(explainer_model: str, non_activating_source: str, 
                  contrastive_config: Tuple[bool, bool, str], train_type: str, 
                  gpu_id: str = "0") -> float:
    """Run a single experiment with the specified configuration."""
    experiment_dir = get_experiment_dir(explainer_model, non_activating_source, contrastive_config, train_type)
    experiment_name = experiment_dir.name
    use_contrastive_explainer, use_contrastive_scorer, contrastive_desc = contrastive_config
    
    print(f"=== Running experiment ===")
    print(f"Explainer model: {explainer_model}")
    print(f"Non-activating source: {non_activating_source}")
    print(f"Contrastive explainer: {use_contrastive_explainer}")
    print(f"Contrastive scorer: {use_contrastive_scorer}")
    print(f"Train type: {train_type}")
    print(f"Experiment name: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Calculate number of GPUs based on GPU IDs
    gpu_ids = [id.strip() for id in gpu_id.split(',') if id.strip()]
    num_gpus = len(gpu_ids)
    
    print(f"Using GPUs: {gpu_ids} (total: {num_gpus})")
    
    # Prepare the command
    cmd = [
        "python", "-m", "delphi",
        BASE_MODEL,
        SPARSE_MODEL,
        "--name", str(experiment_dir.relative_to(Path("results"))),  # Use relative path from results/
        "--hookpoints", HOOKPOINT,
        "--explainer_model", explainer_model,
        "--scorers", "fuzz", "detection",
        "--num_gpus", str(num_gpus),
        "--max_latents", str(MAX_LATENTS),
        "--shared_cache_path", str(get_cache_dir()),
        "--dataset_repo", DATASET_REPO,
        "--dataset_name", DATASET_NAME,
        "--dataset_column", DATASET_COLUMN,
        "--n_tokens", "10000000",
        "--cache_ctx_len", "256",
        "--example_ctx_len", "32",
        "--min_examples", "200",
        "--n_non_activating", "100",
        "--n_examples_train", "40",
        "--n_examples_test", "100",
        "--train_type", train_type,
        "--test_type", "quantiles",
        "--filter_bos",
        "--max_num_seqs", "64", # Needed for larger models to not OOM
        # New configuration options
        "--non_activating_source", non_activating_source,
    ]
    
    # Add contrastive configuration flags
    if use_contrastive_explainer:
        cmd.append("--use_contrastive_explainer")
    if use_contrastive_scorer:
        cmd.append("--use_contrastive_scorer")
    
    # Add thinking mode specific parameters
    if THINKING_MODE:
        cmd.extend([
            "--enable_thinking", "true",
            "--explainer_model_max_len", "131072",
            "--rope_scaling", '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
        ])

    if USE_SEPARATE_SCORER:
        cmd.extend([
            "--scorer_model", "RedHatAI/Qwen3-32B-quantized.w4a16"
        ])

    # Add HF token if available
    if "HF_TOKEN" in os.environ:
        cmd.extend(["--hf_token", os.environ["HF_TOKEN"]])
    
    # Set environment with CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, env=env, check=True, capture_output=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Experiment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
        print()
        
        return duration
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with return code {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        return -1
    except KeyboardInterrupt:
        print("Experiment interrupted by user")
        return -1

def main():
    """Main execution function."""
    # Get GPU ID from environment or use default
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    gpu_ids = [id.strip() for id in gpu_id.split(',') if id.strip()]
    num_gpus = len(gpu_ids)
    
    print("=== Delphi Contrastive Explainer Experiments ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Sparse model: {SPARSE_MODEL}")
    print(f"Max latents: {MAX_LATENTS}")
    print(f"Thinking mode: {THINKING_MODE}")
    print(f"Using GPUs: {gpu_ids} (total: {num_gpus})")
    print(f"Cache directory: {get_cache_dir()}")
    print()
    print("Configuration matrix:")
    print(f"  Non-activating sources: {NON_ACTIVATING_SOURCES}")
    print(f"  Contrastive configs: {[desc for _, _, desc in CONTRASTIVE_CONFIGS]}")
    print(f"  Train types: {TRAIN_TYPES}")
    print(f"  Explainer models: {len(EXPLAINER_MODELS)} models")
    print()
    
    # Check if we're in the right directory
    if not Path("delphi").exists():
        print("ERROR: Please run this script from the delphi-explanations root directory")
        sys.exit(1)
    
    # Setup shared cache
    setup_shared_cache()
    
    # Track results
    results: List[Tuple[str, str, str, str, float]] = []
    
    # Calculate total number of experiments
    total_experiments = (len(EXPLAINER_MODELS) * len(NON_ACTIVATING_SOURCES) * 
                        len(CONTRASTIVE_CONFIGS) * len(TRAIN_TYPES))
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Starting experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    experiment_count = 0
    
    # Run experiments for each configuration combination
    for explainer_model in EXPLAINER_MODELS:
        for non_activating_source in NON_ACTIVATING_SOURCES:
            for contrastive_config in CONTRASTIVE_CONFIGS:
                use_contrastive_explainer, use_contrastive_scorer, contrastive_desc = contrastive_config
                
                # Skip certain combinations that don't make sense
                if non_activating_source == "random" and (use_contrastive_explainer or use_contrastive_scorer):
                    print(f"Skipping {non_activating_source} + {contrastive_desc} (random source with contrastive doesn't make sense)")
                    continue
                
                for train_type in TRAIN_TYPES:
                    experiment_count += 1
                    print(f"Progress: {experiment_count}/{total_experiments}")
                    
                    duration = run_experiment(
                        explainer_model, 
                        non_activating_source, 
                        contrastive_config, 
                        train_type, 
                        gpu_id
                    )
                    
                    results.append((
                        explainer_model, 
                        non_activating_source, 
                        contrastive_desc, 
                        train_type, 
                        duration
                    ))
    
    # Print summary
    print("=== EXPERIMENT SUMMARY ===")
    print(f"All experiments completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Timing Results:")
    print("---------------")
    
    total_time = 0
    for model, source, contrastive, train_type, duration in results:
        if duration > 0:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"{model} | {source} | {contrastive} | {train_type}: {duration:.0f}s ({minutes}m {seconds}s)")
            total_time += duration
        else:
            print(f"{model} | {source} | {contrastive} | {train_type}: FAILED")
    
    if total_time > 0:
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        print()
        print(f"Total execution time: {total_time:.0f}s ({total_minutes}m {total_seconds}s)")
    
    print()
    print("Results saved in:")
    for explainer_model in EXPLAINER_MODELS:
        for non_activating_source in NON_ACTIVATING_SOURCES:
            for contrastive_config in CONTRASTIVE_CONFIGS:
                use_contrastive_explainer, use_contrastive_scorer, contrastive_desc = contrastive_config
                
                # Skip combinations that don't make sense
                if non_activating_source == "random" and (use_contrastive_explainer or use_contrastive_scorer):
                    continue
                    
                for train_type in TRAIN_TYPES:
                    experiment_dir = get_experiment_dir(explainer_model, non_activating_source, contrastive_config, train_type)
                    print(f"  - {experiment_dir}")
    
    print()
    print(f"Shared cache location: {get_cache_dir()}")

if __name__ == "__main__":
    main()
