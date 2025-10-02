#!/bin/bash
# Git bisect test script
# Returns 0 (good) if F1 >= 0.800, returns 1 (bad) if F1 < 0.800

set -e

echo "Testing commit: $(git rev-parse --short HEAD)"

# Run scoring with minimal latents for speed
python -m delphi \
    EleutherAI/pythia-160m \
    EleutherAI/Pythia-160m-SST-k32-32k \
    --hookpoints layers.3.mlp \
    --explainer_model RedHatAI/Qwen3-32B-quantized.w4a16 \
    --scorer_model RedHatAI/Qwen3-32B-quantized.w4a16 \
    --explainer_model_max_len 5120 \
    --explainer_provider offline \
    --explainer default \
    --scorers fuzz \
    --shared_cache_path results/pythiaST/cache \
    --shared_explanations_path "results/pythiaST/Qwen32B Scorer/pythiaST_Qwen3_32B_quantized_w4a16/explanations" \
    --name "pythiaST/Bisect Test/pythiaST_Qwen3_32B_quantized_w4a16" \
    --max_latents 5 \
    --filter_bos true \
    --num_gpus 1 \
    --max_num_seqs 64 \
    --dataset_repo EleutherAI/rpj-v2-sample \
    --dataset_split "train[:1%]" \
    --dataset_name default \
    --dataset_column raw_content \
    --batch_size 32 \
    --cache_ctx_len 256 \
    --n_tokens 10000000 \
    --n_splits 5 \
    --example_ctx_len 32 \
    --min_examples 200 \
    --n_non_activating 100 \
    --center_examples true \
    --non_activating_source random \
    --n_examples_train 40 \
    --n_examples_test 100 \
    --n_quantiles 10 \
    --train_type quantiles \
    --test_type quantiles \
    --ratio_top 0.2 2>&1 | tee bisect_output.log

# Extract F1 score from the output
F1_SCORE=$(grep "Frequency-Weighted F1 Score:" bisect_output.log | tail -1 | awk '{print $5}')

echo "F1 Score: $F1_SCORE"

# Compare using bc for floating point comparison
# Good (0) if F1 >= 0.800, Bad (1) if F1 < 0.800
if (( $(echo "$F1_SCORE >= 0.800" | bc -l) )); then
    echo "✓ GOOD: F1 score $F1_SCORE >= 0.800"
    exit 0
else
    echo "✗ BAD: F1 score $F1_SCORE < 0.800"
    exit 1
fi
