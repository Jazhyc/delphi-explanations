"""
Script to run experiments with a fixed explainer model but varied scorer models.
This allows testing different scoring approaches while reusing explanations.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from typing import List, Tuple

# Import shared configuration
from experiment_config import (
    BASE_MODEL, SPARSE_MODEL, SPARSE_MODEL_NAME, HOOKPOINT,
    DATASET_REPO, DATASET_NAME, DATASET_COLUMN,
    MAX_LATENTS, AVAILABLE_MODELS,
    get_model_name, get_base_dir, get_cache_dir, setup_shared_cache,
    build_base_command
)

# ============================================================================
# VARIED SCORER EXPERIMENT CONFIGURATION
# ============================================================================

DIR_NAME = "Varied Scorers"  # Directory for experiments with varied scorers
THINKING_MODE = False  # Set to True to enable thinking mode

# Fixed explainer model for all experiments
FIXED_EXPLAINER_MODEL = "RedHatAI/Qwen3-32B-quantized.w4a16"

# Non-activating source configurations to test
NON_ACTIVATING_SOURCES = [
    "random", 
]

# Train type configurations to test
TRAIN_TYPES = [
    "quantiles",
]

# Contrastive configurations (optional)
USE_CONTRASTIVE_EXPLAINER = False
USE_CONTRASTIVE_SCORER = False

# Scorer models to test (iterate over these) - uncomment as needed
SCORER_MODELS = [
    # "RedHatAI/gemma-3-4b-it-quantized.w4a16",
    # "RedHatAI/Qwen3-4B-quantized.w4a16",
    # "RedHatAI/gemma-3-12b-it-quantized.w4a16",
    # "RedHatAI/gemma-3-27b-it-quantized.w4a16",
    # "RedHatAI/Qwen3-14B-quantized.w4a16",
    "RedHatAI/Qwen3-32B-quantized.w4a16",
    # "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    # "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_shared_explanations_dir() -> Path:
    """Get the shared explanations directory."""
    explainer_name = get_model_name(FIXED_EXPLAINER_MODEL)
    return get_base_dir() / "explanations" / explainer_name

def get_experiment_dir(scorer_model: str, non_activating_source: str, train_type: str) -> Path:
    """Get the experiment directory for a specific configuration."""
    scorer_name = get_model_name(scorer_model)
    explainer_name = get_model_name(FIXED_EXPLAINER_MODEL)
    
    # Build experiment name components
    components = [
        SPARSE_MODEL_NAME, 
        f"explainer_{explainer_name}",
        f"scorer_{scorer_name}",
        non_activating_source,
        train_type
    ]
    
    if USE_CONTRASTIVE_EXPLAINER or USE_CONTRASTIVE_SCORER:
        contrastive_parts = []
        if USE_CONTRASTIVE_EXPLAINER:
            contrastive_parts.append("ce")
        if USE_CONTRASTIVE_SCORER:
            contrastive_parts.append("cs")
        components.append("_".join(contrastive_parts))
    
    if THINKING_MODE:
        components.append("thinking")
    
    experiment_name = "_".join(components)
    return get_base_dir() / DIR_NAME / experiment_name

def setup_shared_explanations() -> None:
    """Set up shared explanations directory."""
    print("Setting up shared explanations directory...")
    
    explanations_path = get_shared_explanations_dir()
    
    if explanations_path.exists():
        print(f"Shared explanations already exist at {explanations_path}")
        # Check for existing explanation files
        explanation_files = list(explanations_path.glob("*.json"))
        if explanation_files:
            print(f"Found {len(explanation_files)} existing explanation files")
        else:
            print("Explanations directory exists but is empty - will be populated during first run")
    else:
        print(f"Explanations directory does not exist at {explanations_path} - will be created during first run")
    print()

def run_experiment(scorer_model: str, non_activating_source: str, train_type: str, 
                  gpu_id: str = "0") -> float:
    """Run a single experiment with the specified configuration."""
    experiment_dir = get_experiment_dir(scorer_model, non_activating_source, train_type)
    experiment_name = experiment_dir.name
    
    print(f"=== Running experiment ===")
    print(f"Explainer model: {FIXED_EXPLAINER_MODEL}")
    print(f"Scorer model: {scorer_model}")
    print(f"Non-activating source: {non_activating_source}")
    print(f"Train type: {train_type}")
    print(f"Contrastive explainer: {USE_CONTRASTIVE_EXPLAINER}")
    print(f"Contrastive scorer: {USE_CONTRASTIVE_SCORER}")
    print(f"Experiment name: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Calculate number of GPUs based on GPU IDs
    gpu_ids = [id.strip() for id in gpu_id.split(',') if id.strip()]
    num_gpus = len(gpu_ids)
    
    print(f"Using GPUs: {gpu_ids} (total: {num_gpus})")
    
    # Build the command using shared configuration
    cmd = build_base_command(
        experiment_dir=experiment_dir,
        explainer_model=FIXED_EXPLAINER_MODEL,
        num_gpus=num_gpus,
        train_type=train_type,
        non_activating_source=non_activating_source,
        scorer_model=scorer_model,
        shared_explanations_path=get_shared_explanations_dir(),
    )
    
    # Add contrastive configuration flags
    if USE_CONTRASTIVE_EXPLAINER:
        cmd.append("--use_contrastive_explainer")
    if USE_CONTRASTIVE_SCORER:
        cmd.append("--use_contrastive_scorer")
    
    # Add thinking mode specific parameters
    if THINKING_MODE:
        cmd.extend([
            "--enable_thinking", "true",
            "--explainer_model_max_len", "131072",
            "--rope_scaling", '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
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
        print()
        return -1

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Parse GPU ID from command line if provided
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    
    # If multiple GPUs provided, parse them
    gpu_ids = [id.strip() for id in gpu_id.split(',') if id.strip()]
    num_gpus = len(gpu_ids)
    
    print("=== Delphi Varied Scorer Experiments ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Sparse model: {SPARSE_MODEL}")
    print(f"Max latents: {MAX_LATENTS}")
    print(f"Thinking mode: {THINKING_MODE}")
    print(f"Fixed explainer model: {FIXED_EXPLAINER_MODEL}")
    print(f"Contrastive explainer: {USE_CONTRASTIVE_EXPLAINER}")
    print(f"Contrastive scorer: {USE_CONTRASTIVE_SCORER}")
    print(f"Using GPUs: {gpu_ids} (total: {num_gpus})")
    print(f"Cache directory: {get_cache_dir()}")
    print(f"Shared explanations directory: {get_shared_explanations_dir()}")
    print()
    print("Configuration matrix:")
    print(f"  Non-activating sources: {NON_ACTIVATING_SOURCES}")
    print(f"  Train types: {TRAIN_TYPES}")
    print(f"  Scorer models: {len(SCORER_MODELS)} models")
    print()
    
    # Check if we're in the right directory
    if not Path("delphi").exists():
        print("ERROR: Please run this script from the delphi-explanations root directory")
        sys.exit(1)
    
    # Setup shared resources
    setup_shared_cache()
    setup_shared_explanations()
    
    # Track results
    results: List[Tuple[str, str, str, float]] = []
    
    # Calculate total number of experiments
    total_experiments = len(SCORER_MODELS) * len(NON_ACTIVATING_SOURCES) * len(TRAIN_TYPES)
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Starting experiments at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    experiment_count = 0
    
    # Run experiments for each configuration combination
    for scorer_model in SCORER_MODELS:
        for non_activating_source in NON_ACTIVATING_SOURCES:
            for train_type in TRAIN_TYPES:
                experiment_count += 1
                print(f"Progress: {experiment_count}/{total_experiments}")
                
                duration = run_experiment(
                    scorer_model, 
                    non_activating_source, 
                    train_type, 
                    gpu_id
                )
                
                results.append((
                    scorer_model, 
                    non_activating_source, 
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
    for scorer_model, source, train_type, duration in results:
        if duration > 0:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"{scorer_model} | {source} | {train_type}: {duration:.0f}s ({minutes}m {seconds}s)")
            total_time += duration
        else:
            print(f"{scorer_model} | {source} | {train_type}: FAILED")
    
    if total_time > 0:
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        print()
        print(f"Total execution time: {total_time:.0f}s ({total_minutes}m {total_seconds}s)")
    
    print()
    print("Results saved in:")
    for scorer_model in SCORER_MODELS:
        for non_activating_source in NON_ACTIVATING_SOURCES:
            for train_type in TRAIN_TYPES:
                experiment_dir = get_experiment_dir(scorer_model, non_activating_source, train_type)
                print(f"  - {experiment_dir}")
    
    print()
    print(f"Shared cache location: {get_cache_dir()}")
    print(f"Shared explanations location: {get_shared_explanations_dir()}")

if __name__ == "__main__":
    main()
