#!/usr/bin/env python3
"""
Script to run experiments with different explainer models
Based on the millions of papers replication command
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

# Configuration
BASE_MODEL = "google/gemma-2-9b-it"
SPARSE_MODEL = "google/gemma-scope-9b-pt-res"
HOOKPOINT = "layer_32/width_131k/average_l0_51"
DATASET_REPO = "EleutherAI/rpj-v2-sample"
DATASET_NAME = "default"
DATASET_COLUMN = "raw_content"
CACHE_DIR = "results/cache_google_gemma-2-9b-it"
# Cache is organized per layer (e.g., layers.32/) for cleaner structure

# Explainer models to test
EXPLAINER_MODELS = [
    # "RedHatAI/gemma-3-4b-it-quantized.w4a16",
    "RedHatAI/Qwen3-4B-quantized.w4a16",
    # "RedHatAI/SmolLM3-3B-quantized.w4a16"
]

def get_model_name(model_path: str) -> str:
    """Extract a clean model name from the full model path."""
    return model_path.split('/')[-1].replace('-', '_').replace('.', '_')

def setup_shared_cache() -> None:
    """Set up shared activation cache with per-layer organization."""
    print("Setting up shared activation cache...")
    
    cache_path = Path(CACHE_DIR)
    
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

def run_experiment(explainer_model: str, gpu_id: str = "0") -> float:
    """Run a single experiment with the specified explainer model."""
    model_name = get_model_name(explainer_model)
    experiment_name = f"explainer_comparison_{model_name}"
    
    print(f"=== Running experiment with {explainer_model} ===")
    print(f"Experiment name: {experiment_name}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Prepare the command
    cmd = [
        "python", "-m", "delphi",
        BASE_MODEL,
        SPARSE_MODEL,
        "--name", experiment_name,
        "--hookpoints", HOOKPOINT,
        "--explainer_model", explainer_model,
        "--scorers", "fuzz", "detection",
        "--num_gpus", "1",
        "--max_latents", "100",
        "--shared_cache_path", CACHE_DIR,
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
        "--train_type", "random",
        "--test_type", "quantiles",
        "--filter_bos"
    ]
    
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
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
    
    print("=== Delphi Explainer Model Comparison Experiments ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Sparse model: {SPARSE_MODEL}")
    print(f"Using GPU: {gpu_id}")
    print(f"Cache directory: {CACHE_DIR}")
    print()
    
    # Check if we're in the right directory
    if not Path("delphi").exists():
        print("ERROR: Please run this script from the delphi-explanations root directory")
        sys.exit(1)
    
    # Setup shared cache
    setup_shared_cache()
    
    # Track results
    results: List[Tuple[str, float]] = []
    
    print(f"Starting experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run experiments for each explainer model
    for i, explainer_model in enumerate(EXPLAINER_MODELS, 1):
        print(f"Progress: {i}/{len(EXPLAINER_MODELS)}")
        duration = run_experiment(explainer_model, gpu_id)
        results.append((explainer_model, duration))
    
    # Print summary
    print("=== EXPERIMENT SUMMARY ===")
    print(f"All experiments completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Timing Results:")
    print("---------------")
    
    total_time = 0
    for model, duration in results:
        if duration > 0:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"{model}: {duration:.0f}s ({minutes}m {seconds}s)")
            total_time += duration
        else:
            print(f"{model}: FAILED")
    
    if total_time > 0:
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        print()
        print(f"Total execution time: {total_time:.0f}s ({total_minutes}m {total_seconds}s)")
    
    print()
    print("Results saved in:")
    for explainer_model in EXPLAINER_MODELS:
        model_name = get_model_name(explainer_model)
        print(f"  - results/explainer_comparison_{model_name}/")
    
    print()
    print(f"Shared cache location: {CACHE_DIR}")

if __name__ == "__main__":
    main()
