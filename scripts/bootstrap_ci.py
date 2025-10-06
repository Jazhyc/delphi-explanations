"""
Bootstrap confidence interval computation for frequency-weighted F1 scores.

This module provides optimized parallel bootstrap resampling for computing
confidence intervals on frequency-weighted F1 scores across latent features.
"""

import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def compute_weighted_ci_errors(score_subset, counts, freq_weighted_f1, confidence, n_boot=1000, n_jobs=8):
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
        
    Returns:
        Tuple of (lower_error, upper_error) for confidence interval
    """
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
    
    return lower_error, upper_error
