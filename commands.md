# Testing commands

## Quick debug

python -m delphi EleutherAI/pythia-160m EleutherAI/sae-pythia-160m-32k \
  --explainer_model Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --n_tokens 200000 \
  --max_latents 10 \
  --hookpoints layers.3.mlp \
  --scorers detection \
  --name pythia-160m-test \
  --explainer_model_max_len 3192 \
  --max_num_seqs 64 \
  --num_gpus 2
  --name Debug

## Replicate results from millions of papers

python -m delphi \
    google/gemma-2-9b-it \
    google/gemma-scope-9b-pt-res \
    --name "qwen-exp1-uniform-sampling" \
    --hookpoints "layer_32/width_131k/average_l0_51" \
    --explainer_model "Qwen/Qwen3-30B-A3B-GPTQ-Int4" \
    --scorers fuzz detection \
    --num_gpus 2 \
    --max_num_seqs 64 \
    --dataset_repo "EleutherAI/rpj-v2-sample" \
    --dataset_name "default" \
    --dataset_column "raw_content" \
    --n_tokens 10000000 \
    --cache_ctx_len 256 \
    --example_ctx_len 32 \
    --min_examples 200 \
    --n_non_activating 100 \
    --n_examples_train 40 \
    --n_examples_test 100 \
    --train_type "random" \
    --test_type "quantiles" \
    --hf_token "$HF_TOKEN" \
    --filter_bos