#!/usr/bin/env python3
"""
Script to check the F1 score discrepancy between terminal output and plots.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the parent directory to the path to import delphi modules
sys.path.append(str(Path.cwd()))

from delphi.log.result_analysis import (
    load_data,
    get_agg_metrics,
    add_latent_f1,
)

# Test with one experiment
experiment_dir = Path("results/pythiaST/Varied Scorers/pythiaST_explainer_Qwen3_32B_quantized_w4a16_scorer_Qwen3_32B_quantized_w4a16_random_quantiles")

if not experiment_dir.exists():
    print(f"Experiment directory not found: {experiment_dir}")
    sys.exit(1)

scores_path = experiment_dir / "scores"
latents_path = Path("results/pythiaST/cache/latents")

# Extract module name from score files
sample_score_dir = next(scores_path.iterdir())
sample_files = list(sample_score_dir.glob("*.txt"))
if sample_files:
    sample_filename = sample_files[0].stem
    module_name = sample_filename.split('_latent')[0]
    modules = [module_name]
    print(f"Module: {module_name}")
else:
    print("No score files found")
    sys.exit(1)

# Load data using the same method as the notebook
print("\n=== Loading data (same as notebook) ===")
latent_df, counts = load_data(scores_path, latents_path, modules)
print(f"Loaded {len(latent_df)} rows")
print(f"Unique latents: {latent_df[['module', 'latent_idx']].drop_duplicates().shape[0]}")

# Add F1 scores
latent_df = add_latent_f1(latent_df)

# Get aggregated metrics (same as notebook)
processed_df = get_agg_metrics(latent_df, counts)

print("\n=== Processed DF (what notebook uses) ===")
print(processed_df[['score_type', 'accuracy', 'f1_score', 'precision', 'recall', 'weighted_f1']])

print("\n=== Terminal Output Format (from log_results) ===")
for score_type in processed_df.score_type.unique():
    score_type_summary = processed_df[processed_df.score_type == score_type].iloc[0]
    print(f"\n--- {score_type.title()} Metrics ---")
    print(f"Class-Balanced Accuracy: {score_type_summary['accuracy']:.3f}")
    print(f"F1 Score: {score_type_summary['f1_score']:.3f}")
    print(f"Frequency-Weighted F1 Score: {score_type_summary['weighted_f1']:.3f}")
    print(f"Precision: {score_type_summary['precision']:.3f}")
    print(f"Recall: {score_type_summary['recall']:.3f}")

# Now check what the notebook computes for mean
print("\n=== What Notebook Uses for Pareto Plot ===")
fuzz_row = processed_df[processed_df['score_type'] == 'fuzz']
detection_row = processed_df[processed_df['score_type'] == 'detection']

if len(fuzz_row) > 0 and len(detection_row) > 0:
    fuzz_f1 = fuzz_row['weighted_f1'].iloc[0]
    detection_f1 = detection_row['weighted_f1'].iloc[0]
    mean_f1 = (fuzz_f1 + detection_f1) / 2.0
    
    print(f"Fuzz weighted_f1: {fuzz_f1:.3f}")
    print(f"Detection weighted_f1: {detection_f1:.3f}")
    print(f"Mean (used in plot): {mean_f1:.3f}")
