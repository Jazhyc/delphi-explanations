"""
Bootstrap confidence interval computation for frequency-weighted F1 scores.

This module provides optimized parallel bootstrap resampling for computing
confidence intervals on frequency-weighted F1 scores across latent features.
Also includes random baseline computation and caching support.
"""

import numpy as np
import hashlib
import json
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Cache directory for storing bootstrap and baseline results
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _single_bootstrap_iteration(seed, latent_data, latent_weights):
    """
    Compute one bootstrap sample. This function is designed to be called in parallel.
    
    Args:
        seed: Random seed for reproducibility
        latent_data: List of dicts with 'predictions', 'labels', 'n_examples'
        latent_weights: Array of firing frequency weights
        
    Returns:
        float: Frequency-weighted F1 for this bootstrap sample
    """
    np.random.seed(seed)
    latent_f1s_boot = []
    
    for latent in latent_data:
        n_ex = latent['n_examples']
        # Resample indices for this latent
        boot_idx = np.random.randint(0, n_ex, size=n_ex)
        
        # Get bootstrapped predictions and labels
        boot_preds = latent['predictions'][boot_idx]
        boot_labels = latent['labels'][boot_idx]
        
        # Compute confusion matrix elements
        tp = np.sum((boot_preds == 1) & (boot_labels == 1))
        fp = np.sum((boot_preds == 1) & (boot_labels == 0))
        tn = np.sum((boot_preds == 0) & (boot_labels == 0))
        fn = np.sum((boot_preds == 0) & (boot_labels == 1))
        
        # Compute F1 directly from confusion matrix
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        latent_f1s_boot.append(f1)
    
    # Compute frequency-weighted F1 for this bootstrap sample
    latent_f1s_boot = np.asarray(latent_f1s_boot, dtype=np.float64)
    weighted_f1_boot = (latent_f1s_boot * latent_weights).sum() / latent_weights.sum()
    return float(weighted_f1_boot)


def _compute_cache_key(score_subset, counts, n_boot, confidence, operation="bootstrap"):
    """
    Compute a cache key based on the data and parameters.
    
    Args:
        score_subset: DataFrame with score data
        counts: Dictionary mapping module names to firing count tensors
        n_boot: Number of bootstrap/random samples
        confidence: Confidence level
        operation: Type of operation ("bootstrap" or "random_baseline")
        
    Returns:
        Tuple of (readable_key, hash_key)
    """
    # Create a deterministic representation of the data
    score_type = str(score_subset['score_type'].iloc[0]) if len(score_subset) > 0 else 'unknown'
    n_latents = len(score_subset.groupby(["module", "latent_idx"]))
    
    cache_data = {
        'operation': operation,
        'n_boot': n_boot,
        'confidence': confidence,
        'n_examples': len(score_subset),
        'score_type': score_type,
    }
    
    # Add latent-level summary
    latent_summary = []
    for (module, latent_idx), grp in score_subset.groupby(["module", "latent_idx"]):
        if module in counts and latent_idx < len(counts[module]):
            fire_count = counts[module][latent_idx].item()
            n_ex = len(grp)
            # Add predictions and labels for uniqueness
            pred_sum = grp['prediction'].sum() if 'prediction' in grp.columns else 0
            label_sum = grp['activating'].sum() if 'activating' in grp.columns else 0
            latent_summary.append((str(module), int(latent_idx), int(fire_count), int(n_ex), int(pred_sum), int(label_sum)))
    
    cache_data['latent_summary'] = sorted(latent_summary)
    
    # Create hash for uniqueness
    cache_str = json.dumps(cache_data, sort_keys=True)
    cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    # Create readable key
    readable_key = f"{operation}_{score_type}_{n_latents}latents_{n_boot}samples_conf{int(confidence*100)}"
    
    return readable_key, cache_hash


def _load_from_cache(readable_key, cache_hash):
    """Load cached results if available."""
    cache_file = CACHE_DIR / f"{readable_key}_{cache_hash[:8]}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    return None


def _save_to_cache(readable_key, cache_hash, data):
    """Save results to cache with readable filename."""
    cache_file = CACHE_DIR / f"{readable_key}_{cache_hash[:8]}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def compute_weighted_ci_errors(score_subset, counts, freq_weighted_f1, confidence, n_boot=1000, n_jobs=8, use_cache=True):
    """
    Compute confidence interval errors using optimized bootstrap resampling with multiprocessing.
    
    For single-module cases, we resample examples within each latent while
    respecting the hierarchical structure. Latents contribute to the final
    metric proportional to their firing frequency.
    
    Optimized by pre-computing per-latent confusion matrices, using
    vectorized operations, and parallel processing across bootstrap samples.
    
    Args:
        score_subset: DataFrame with score data for a specific score type
        counts: Dictionary mapping module names to firing count tensors
        freq_weighted_f1: Observed frequency-weighted F1 score
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_boot: Number of bootstrap resamples
        n_jobs: Number of parallel processes to use
        use_cache: Whether to use cached results if available
        
    Returns:
        Tuple of (lower_error, upper_error) for confidence interval
    """
    # Check cache first and compute cache keys
    readable_key = None
    cache_hash = None
    if use_cache:
        readable_key, cache_hash = _compute_cache_key(score_subset, counts, n_boot, confidence, "bootstrap")
        cached_result = _load_from_cache(readable_key, cache_hash)
        if cached_result is not None:
            print(f"  [Using cached bootstrap results: {readable_key}]")
            return cached_result['lower_error'], cached_result['upper_error']
    
    # Pre-compute per-latent confusion matrices and metadata
    latent_data = []
    latent_weights = []
    
    for (module, latent_idx), grp in score_subset.groupby(["module", "latent_idx"]):
        if module not in counts or latent_idx >= len(counts[module]):
            continue
            
        fire_count = counts[module][latent_idx].item()
        
        # Store predictions and labels for this latent
        if len(grp) == 0:
            continue
        
        # Check for required columns
        if 'prediction' not in grp.columns:
            raise KeyError(f"Cannot find 'prediction' column. Available columns: {list(grp.columns)}")
        
        if 'activating' not in grp.columns:
            raise KeyError(f"Cannot find 'activating' column (true label). Available columns: {list(grp.columns)}")
            
        latent_data.append({
            'predictions': grp['prediction'].values,
            'labels': grp['activating'].values,  # activating is the true label
            'n_examples': len(grp)
        })
        latent_weights.append(float(fire_count))
    
    if not latent_data:
        return 0.0, 0.0
    
    latent_weights = np.asarray(latent_weights, dtype=np.float64)
    
    # Generate random seeds for reproducibility
    base_seed = np.random.randint(0, 2**31)
    seeds = [base_seed + i for i in range(n_boot)]
    
    # Run bootstrap iterations in parallel
    boot_samples = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all bootstrap iterations
        future_to_seed = {
            executor.submit(_single_bootstrap_iteration, seed, latent_data, latent_weights): seed 
            for seed in seeds
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_seed), total=n_boot, desc="Bootstrap CI", leave=False):
            try:
                boot_samples.append(future.result())
            except Exception as e:
                print(f"Bootstrap iteration failed: {e}")
                continue
    
    if not boot_samples:
        return 0.0, 0.0
    
    # Compute confidence interval from bootstrap distribution
    lower_pct = (1.0 - confidence) / 2.0 * 100.0
    upper_pct = (1.0 + confidence) / 2.0 * 100.0
    lower_ci, upper_ci = np.percentile(boot_samples, [lower_pct, upper_pct])
    
    # Return errors from the observed value
    lower_error = float(freq_weighted_f1 - lower_ci)
    upper_error = float(upper_ci - freq_weighted_f1)
    
    # Save to cache
    if use_cache:
        cache_result = {
            'lower_error': lower_error,
            'upper_error': upper_error,
            'n_boot': n_boot,
            'confidence': confidence
        }
        _save_to_cache(readable_key, cache_hash, cache_result)
    
    return lower_error, upper_error


def _single_random_iteration(seed, latent_data, latent_weights, use_weights=False):
    """
    Compute one random baseline sample where predictions are randomized.
    
    Args:
        seed: Random seed for reproducibility
        latent_data: List of dicts with 'predictions', 'labels', 'n_examples'
        latent_weights: Array of firing frequency weights
        use_weights: If True, compute frequency-weighted F1; if False, compute unweighted F1
        
    Returns:
        float: F1 score for random predictions (weighted or unweighted)
    """
    np.random.seed(seed)
    
    if use_weights:
        # Compute frequency-weighted F1: compute per-latent F1 and weight by firing frequency
        latent_f1s_random = []
        
        for latent in latent_data:
            n_ex = latent['n_examples']
            labels = latent['labels']
            
            # Generate random predictions (50/50 chance for each example)
            random_preds = np.random.randint(0, 2, size=n_ex)
            
            # Compute confusion matrix elements
            tp = np.sum((random_preds == 1) & (labels == 1))
            fp = np.sum((random_preds == 1) & (labels == 0))
            tn = np.sum((random_preds == 0) & (labels == 0))
            fn = np.sum((random_preds == 0) & (labels == 1))
            
            # Compute F1 directly from confusion matrix
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            latent_f1s_random.append(f1)
        
        latent_f1s_random = np.asarray(latent_f1s_random, dtype=np.float64)
        weighted_f1_random = (latent_f1s_random * latent_weights).sum() / latent_weights.sum()
        return float(weighted_f1_random)
    else:
        # Compute unweighted F1: aggregate all examples together and compute F1 globally
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for latent in latent_data:
            n_ex = latent['n_examples']
            labels = latent['labels']
            
            # Generate random predictions (50/50 chance for each example)
            random_preds = np.random.randint(0, 2, size=n_ex)
            
            # Accumulate confusion matrix elements
            total_tp += np.sum((random_preds == 1) & (labels == 1))
            total_fp += np.sum((random_preds == 1) & (labels == 0))
            total_fn += np.sum((random_preds == 0) & (labels == 1))
        
        # Compute global F1 from accumulated confusion matrix
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return float(f1)


def compute_random_baseline(score_subset, counts, n_samples=100, n_jobs=8, use_cache=True, use_weights=False):
    """
    Compute random baseline by generating random predictions for each example.
    
    The baseline represents the expected performance if predictions were made
    randomly (50/50 chance for each example). Can optionally weight by firing frequency.
    
    Args:
        score_subset: DataFrame with score data for a specific score type
        counts: Dictionary mapping module names to firing count tensors
        n_samples: Number of random samples to average over
        n_jobs: Number of parallel processes to use
        use_cache: Whether to use cached results if available
        use_weights: If True, compute frequency-weighted F1; if False, compute unweighted F1
        
    Returns:
        float: Mean random baseline F1 score
    """
    # Check cache first and compute cache keys
    readable_key = None
    cache_hash = None
    if use_cache:
        # Include use_weights in cache key to distinguish weighted vs unweighted baselines
        operation_name = "random_baseline_weighted" if use_weights else "random_baseline_unweighted"
        readable_key, cache_hash = _compute_cache_key(score_subset, counts, n_samples, 0.0, operation_name)
        cached_result = _load_from_cache(readable_key, cache_hash)
        if cached_result is not None:
            print(f"  [Using cached random baseline: {readable_key}]")
            return cached_result['baseline_f1']
    
    # Pre-compute per-latent data
    latent_data = []
    latent_weights = []
    
    for (module, latent_idx), grp in score_subset.groupby(["module", "latent_idx"]):
        if module not in counts or latent_idx >= len(counts[module]):
            continue
            
        fire_count = counts[module][latent_idx].item()
        
        if len(grp) == 0:
            continue
        
        # Check for required columns
        if 'activating' not in grp.columns:
            raise KeyError(f"Cannot find 'activating' column (true label). Available columns: {list(grp.columns)}")
            
        latent_data.append({
            'labels': grp['activating'].values,
            'n_examples': len(grp)
        })
        latent_weights.append(float(fire_count))
    
    if not latent_data:
        return 0.0
    
    latent_weights = np.asarray(latent_weights, dtype=np.float64)
    
    # Generate random seeds for reproducibility
    base_seed = np.random.randint(0, 2**31)
    seeds = [base_seed + i for i in range(n_samples)]
    
    # Run random baseline iterations in parallel
    random_samples = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_seed = {
            executor.submit(_single_random_iteration, seed, latent_data, latent_weights, use_weights): seed 
            for seed in seeds
        }
        
        for future in tqdm(as_completed(future_to_seed), total=n_samples, desc="Random baseline", leave=False):
            try:
                random_samples.append(future.result())
            except Exception as e:
                print(f"Random baseline iteration failed: {e}")
                continue
    
    if not random_samples:
        return 0.0
    
    # Compute mean random baseline
    baseline_f1 = float(np.mean(random_samples))
    
    # Save to cache
    if use_cache:
        cache_result = {
            'baseline_f1': baseline_f1,
            'n_samples': n_samples,
            'std': float(np.std(random_samples))
        }
        _save_to_cache(readable_key, cache_hash, cache_result)
    
    return baseline_f1
