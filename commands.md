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