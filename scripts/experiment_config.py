"""
Shared configuration for experiment scripts.
This module contains common settings, hyperparameters, and utility functions
used across different experiment runner scripts.
"""

from pathlib import Path
from typing import Tuple

# ============================================================================
# BASE MODEL CONFIGURATION
# ============================================================================

BASE_MODEL = "EleutherAI/pythia-160m"
SPARSE_MODEL = "EleutherAI/Pythia-160m-SST-k32-32k"
SPARSE_MODEL_NAME = "pythiaST"  # Short name for directory structure
HOOKPOINT = "layers.3.mlp"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_REPO = "EleutherAI/rpj-v2-sample"
DATASET_NAME = "default"
DATASET_COLUMN = "raw_content"

# ============================================================================
# EXPERIMENT HYPERPARAMETERS
# ============================================================================

MAX_LATENTS = 100
N_TOKENS = "10000000"
CACHE_CTX_LEN = "256"
EXAMPLE_CTX_LEN = "32"
MIN_EXAMPLES = "200"
N_NON_ACTIVATING = "100"
N_EXAMPLES_TRAIN = "40"
N_EXAMPLES_TEST = "100"
TEST_TYPE = "quantiles"
MAX_NUM_SEQS = "64"  # Needed for larger models to not OOM
SCORERS = ["fuzz", "detection"] # detection

# ============================================================================
# MODEL LISTS
# ============================================================================

# Available models for explainers and scorers
AVAILABLE_MODELS = [
    "RedHatAI/gemma-3-4b-it-quantized.w4a16",
    "RedHatAI/Qwen3-4B-quantized.w4a16",
    "RedHatAI/gemma-3-12b-it-quantized.w4a16",
    "RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "RedHatAI/Qwen3-14B-quantized.w4a16",
    "RedHatAI/Qwen3-32B-quantized.w4a16",
    "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    # "Transluce/llama_8b_explainer",
    # "openai/gpt-oss-20b",
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_name(model_path: str) -> str:
    """Extract a clean model name from the full model path."""
    return model_path.split('/')[-1].replace('-', '_').replace('.', '_')

def get_base_dir() -> Path:
    """Get the base directory for this sparse model."""
    return Path("results") / SPARSE_MODEL_NAME

def get_cache_dir() -> Path:
    """Get the shared cache directory."""
    return get_base_dir() / "cache"

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

def build_base_command(
    experiment_dir: Path,
    explainer_model: str,
    num_gpus: int,
    train_type: str,
    non_activating_source: str,
    scorer_model: str | None = None,
    shared_explanations_path: Path | None = None,
) -> list:
    """Build the base delphi command with common arguments."""
    
    cmd = [
        "python", "-m", "delphi",
        BASE_MODEL,
        SPARSE_MODEL,
        "--name", str(experiment_dir.relative_to(Path("results"))),
        "--hookpoints", HOOKPOINT,
        "--explainer_model", explainer_model,
        "--scorers", *SCORERS,
        "--num_gpus", str(num_gpus),
        "--max_latents", str(MAX_LATENTS),
        "--shared_cache_path", str(get_cache_dir()),
        "--dataset_repo", DATASET_REPO,
        "--dataset_name", DATASET_NAME,
        "--dataset_column", DATASET_COLUMN,
        "--n_tokens", N_TOKENS,
        "--cache_ctx_len", CACHE_CTX_LEN,
        "--example_ctx_len", EXAMPLE_CTX_LEN,
        "--min_examples", MIN_EXAMPLES,
        "--n_non_activating", N_NON_ACTIVATING,
        "--n_examples_train", N_EXAMPLES_TRAIN,
        "--n_examples_test", N_EXAMPLES_TEST,
        "--train_type", train_type,
        "--test_type", TEST_TYPE,
        "--filter_bos",
        "--max_num_seqs", MAX_NUM_SEQS,
        "--non_activating_source", non_activating_source,
    ]
    
    # Add scorer model if different from explainer
    if scorer_model and scorer_model != explainer_model:
        cmd.extend(["--scorer_model", scorer_model])
    
    # Add shared explanations path if provided
    if shared_explanations_path:
        cmd.extend(["--shared_explanations_path", str(shared_explanations_path)])
    
    return cmd
