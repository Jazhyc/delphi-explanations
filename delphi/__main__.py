import asyncio
import logging
import os
import gc
import time
from functools import partial
from pathlib import Path
from typing import Callable

import orjson
import torch
from simple_parsing import ArgumentParser
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi import logger
from delphi.clients import Offline, OpenRouter
from delphi.config import RunConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer, NoOpExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer, OpenAISimulator
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import assert_type, load_tokenized_data


# ============================================================================
# Shared Helper Functions for Scoring Pipeline
# ============================================================================

def create_scorer_preprocess():
    """
    Create a preprocessor function for scorers.
    This function prepares LatentRecords for scoring by attaching explanations
    and non-activating examples.
    
    Returns:
        A function that preprocesses results for scoring.
    """
    def scorer_preprocess(result):
        if isinstance(result, list):
            result = result[0]
        record = result.record
        record.explanation = result.explanation
        # Always provide non-activating examples - scorers that don't need them will ignore them
        record.extra_examples = record.not_active
        return record
    return scorer_preprocess


def create_scorer_postprocess(score_dir: Path):
    """
    Create a postprocessor function for scorers.
    This function saves scoring results to disk.
    
    Args:
        score_dir: Directory where score files should be saved.
        
    Returns:
        A function that saves scorer results to files.
    """
    def scorer_postprocess(result, score_dir):
        # Replace "/" with "--" to avoid directory separators in filenames
        safe_latent_name = str(result.record.latent).replace("/", "--")
        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))
    return scorer_postprocess


# ============================================================================
# Model and Artifact Loading
# ============================================================================

def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        run_cfg.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )

    return (
        list(hookpoint_to_sparse_encode.keys()),
        hookpoint_to_sparse_encode,
        model,
        transcode,
    )


def create_neighbours(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    constructor_cfg = run_cfg.constructor_cfg
    neighbour_type = constructor_cfg.non_activating_source

    saes = (
        load_sparse_coders(run_cfg, device="cpu")
        if neighbour_type != "co-occurrence"
        else {}
    )

    for hookpoint in hookpoints:

        if neighbour_type == "co-occurrence":
            neighbour_calculator = NeighbourCalculator(
                cache_dir=latents_path / hookpoint, number_of_neighbours=250
            )

        elif neighbour_type == "decoder_similarity":

            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )

        elif neighbour_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {neighbour_type} not supported for neighbour creation."
            )

        neighbour_calculator.populate_neighbour_cache(neighbour_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def generate_explanations(
    run_cfg: RunConfig,
    latents_path: Path,
    explanations_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
    scorer_model_name: str | None = None,
) -> tuple[object | None, float]:
    """Stage 1: Loads the explainer model, generates explanations for all latents,
    and saves them to disk before unloading the model.
    Returns a tuple of (client_or_none, explanation_time_seconds).
    """
    print("--- Starting Stage 1: Explanation Generation ---")
    explanations_path.mkdir(parents=True, exist_ok=True)

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {hook: latent_range for hook in hookpoints}

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
        load_neighbours=(
            run_cfg.use_contrastive_explainer 
            and run_cfg.constructor_cfg.non_activating_source in ["co-occurrence", "decoder_similarity", "encoder_similarity"]
        ),
        generate_non_activating=(
            run_cfg.use_contrastive_explainer 
            or scorers_need_non_activating_examples(run_cfg.scorers)
        ),
    )

    if run_cfg.explainer == "none":
        print("Explainer set to 'none' - skipping explanation generation stage.")
        return None, 0.0

    # Initialize explainer LLM client
    if run_cfg.explainer_provider == "offline":
        llm_client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
            max_num_seqs=run_cfg.max_num_seqs,
            expert_parallel=run_cfg.enable_expert_parallel,
            enable_thinking=run_cfg.enable_thinking,
            rope_scaling=run_cfg.rope_scaling_dict,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set `--explainer-provider offline` to use a local explainer model."
            )
        llm_client = OpenRouter(run_cfg.explainer_model, api_key=os.environ["OPENROUTER_API_KEY"])  # type: ignore
    else:
        raise ValueError(f"Explainer provider {run_cfg.explainer_provider} not supported")

    def explainer_postprocess(result):
        with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))
        return result

    if run_cfg.use_contrastive_explainer:
        explainer = ContrastiveExplainer(llm_client, threshold=0.3, verbose=run_cfg.verbose)
    else:
        explainer = DefaultExplainer(llm_client, threshold=0.3, verbose=run_cfg.verbose)

    explainer_pipe = Pipe(process_wrapper(explainer, postprocess=explainer_postprocess))

    pipeline = Pipeline(dataset, explainer_pipe, progress_description="Generating explanations")
    
    # Time the pipeline execution
    explainer_start_time = time.time()
    await pipeline.run(run_cfg.pipeline_num_proc)
    explainer_end_time = time.time()

    # Calculate explanation time
    explanation_time = explainer_end_time - explainer_start_time

    # Save explainer stats
    stats_path = explanations_path.parent / "explainer_stats.json"
    stats_to_save = {key: dict(value) for key, value in explainer.stats.items()}
    with open(stats_path, "wb") as f:
        f.write(orjson.dumps(stats_to_save, option=orjson.OPT_INDENT_2))

    # If the scorer model requested is the same as the explainer model, keep the
    # explainer client live and return it so the caller can reuse it and avoid
    # recompilation. Otherwise unload the client.
    if scorer_model_name is not None and scorer_model_name == run_cfg.explainer_model:
        print("Explainer and scorer model identical — keeping model loaded to avoid recompilation.")
        return llm_client, explanation_time

    # Unload the model
    close_fn = getattr(llm_client, "close", None)
    if callable(close_fn):
        # close may be async or sync
        if asyncio.iscoroutinefunction(close_fn):
            await close_fn()
        else:
            close_fn()
    del llm_client
    del explainer
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    print("--- Finished Stage 1: Explainer model unloaded. ---")
    return None, explanation_time


async def run_scoring(
    run_cfg: RunConfig,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
    existing_llm_client: object | None = None,
) -> float:
    """Stage 2: Loads the scorer model, reads explanations from disk, runs all scorers,
    and saves the scores before unloading the model.
    Returns the scoring time in seconds.
    """
    print("--- Starting Stage 2: Scoring ---")

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {hook: latent_range for hook in hookpoints}

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
        neighbours_path=neighbours_path if neighbours_path.exists() else None,
        load_neighbours=(
            run_cfg.use_contrastive_scorer 
            and run_cfg.constructor_cfg.non_activating_source in ["co-occurrence", "decoder_similarity", "encoder_similarity"]
        ),
        generate_non_activating=(
            run_cfg.use_contrastive_scorer 
            or scorers_need_non_activating_examples(run_cfg.scorers)
        ),
    )

    # Determine scorer model name (fallback to explainer model)
    scorer_model_name = run_cfg.scorer_model if getattr(run_cfg, "scorer_model", None) else run_cfg.explainer_model
    print(f"Loading scorer model: {scorer_model_name}")

    # Use existing client if provided (e.g., same model reused); otherwise initialize
    if existing_llm_client is not None:
        scorer_llm_client = existing_llm_client
    else:
        # Initialize scorer LLM client
        if run_cfg.explainer_provider == "offline":
            scorer_llm_client = Offline(
                scorer_model_name,
                max_memory=0.9,
                max_model_len=run_cfg.explainer_model_max_len,
                num_gpus=run_cfg.num_gpus,
                statistics=run_cfg.verbose,
                max_num_seqs=run_cfg.max_num_seqs,
                expert_parallel=run_cfg.enable_expert_parallel,
                enable_thinking=run_cfg.enable_thinking,
                rope_scaling=run_cfg.rope_scaling_dict,
            )
        elif run_cfg.explainer_provider == "openrouter":
            if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable not set. Set `--explainer-provider offline` to use a local explainer model."
                )
            scorer_llm_client = OpenRouter(scorer_model_name, api_key=os.environ["OPENROUTER_API_KEY"])  # type: ignore
        else:
            raise ValueError(f"Explainer provider {run_cfg.explainer_provider} not supported")

    # NoOp explainer loads explanations from disk
    def none_postprocessor(result):
        explanation_path = explanations_path / f"{result.record.latent}.txt"
        if not explanation_path.exists():
            raise FileNotFoundError(f"Explanation file {explanation_path} does not exist.")
        with open(explanation_path, "rb") as f:
            return ExplainerResult(record=result.record, explanation=orjson.loads(f.read()))

    explainer_pipe = Pipe(process_wrapper(NoOpExplainer(), postprocess=none_postprocessor))

    # Use shared scorer preprocess/postprocess functions
    scorer_preprocess = create_scorer_preprocess()

    scorers = []
    
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)
        stats_path = scorer_path.parent.parent / f"{scorer_name}_stats.json"

        if scorer_name == "simulation":
            scorer = OpenAISimulator(scorer_llm_client, tokenizer=tokenizer, all_at_once=False)
        elif scorer_name == "fuzz":
            scorer = FuzzingScorer(
                scorer_llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
                stats_path=stats_path,
            )
        elif scorer_name == "detection":
            scorer = DetectionScorer(
                scorer_llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
                stats_path=stats_path,
            )
        else:
            raise ValueError(f"Scorer {scorer_name} not supported")

        wrapped_scorer = process_wrapper(
            scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(create_scorer_postprocess(scorer_path), score_dir=scorer_path),
        )
        scorers.append(wrapped_scorer)

    pipeline = Pipeline(dataset, explainer_pipe, Pipe(*scorers), progress_description="Scoring explanations")
    if run_cfg.pipeline_num_proc > 1 and run_cfg.explainer_provider == "openrouter":
        print("OpenRouter does not support multiprocessing, setting pipeline_num_proc to 1")
        run_cfg.pipeline_num_proc = 1

    # Time the pipeline execution
    pipeline_start_time = time.time()
    await pipeline.run(run_cfg.pipeline_num_proc)
    pipeline_end_time = time.time()
    
    # Calculate scoring time
    scoring_time = pipeline_end_time - pipeline_start_time

    # Unload scorer model
    close_fn = getattr(scorer_llm_client, "close", None)
    if callable(close_fn):
        if asyncio.iscoroutinefunction(close_fn):
            await close_fn()
        else:
            close_fn()
    del scorer_llm_client
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    print("--- Finished Stage 2: Scorer model unloaded. ---")
    return scoring_time


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
        load_neighbours=(
            (run_cfg.use_contrastive_explainer or run_cfg.use_contrastive_scorer)
            and run_cfg.constructor_cfg.non_activating_source in ["co-occurrence", "decoder_similarity", "encoder_similarity"]
        ),
        generate_non_activating=(
            run_cfg.use_contrastive_explainer 
            or run_cfg.use_contrastive_scorer 
            or scorers_need_non_activating_examples(run_cfg.scorers)
        ),
    )
    
    def create_llm_client(model_name: str):
        if run_cfg.explainer_provider == "offline":
            return Offline(
                model_name,
                max_memory=0.9,
                max_model_len=run_cfg.explainer_model_max_len,
                num_gpus=run_cfg.num_gpus,
                statistics=run_cfg.verbose,
                max_num_seqs=run_cfg.max_num_seqs,
                expert_parallel=run_cfg.enable_expert_parallel,
                enable_thinking=run_cfg.enable_thinking,
                rope_scaling=run_cfg.rope_scaling_dict,
            )
        elif run_cfg.explainer_provider == "openrouter":
            if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable not set. Set "
                    "`--explainer-provider offline` to use a local explainer model."
                )
            return OpenRouter(
                model_name,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            raise ValueError(
                f"Explainer provider {run_cfg.explainer_provider} not supported"
            )

    if run_cfg.explainer_provider == "offline":
        llm_client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
            max_num_seqs=run_cfg.max_num_seqs,
            expert_parallel=run_cfg.enable_expert_parallel,
            enable_thinking=run_cfg.enable_thinking,
            rope_scaling=run_cfg.rope_scaling_dict,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )

        llm_client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    if not run_cfg.explainer == "none":

        def explainer_postprocess(result):
            with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
                f.write(orjson.dumps(result.explanation))

            return result

        if run_cfg.use_contrastive_explainer:
            explainer = ContrastiveExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )
        else:
            explainer = DefaultExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
            )

        explainer_pipe = Pipe(
            process_wrapper(explainer, postprocess=explainer_postprocess)
        )
    else:

        def none_postprocessor(result):
            # Load the explanation from disk
            explanation_path = explanations_path / f"{result.record.latent}.txt"
            if not explanation_path.exists():
                raise FileNotFoundError(
                    f"Explanation file {explanation_path} does not exist. "
                    "Make sure to run an explainer pipeline first."
                )

            with open(explanation_path, "rb") as f:
                return ExplainerResult(
                    record=result.record,
                    explanation=orjson.loads(f.read()),
                )

        explainer_pipe = Pipe(
            process_wrapper(
                NoOpExplainer(),
                postprocess=none_postprocessor,
            )
        )

    # Use shared scorer preprocess/postprocess functions
    scorer_preprocess = create_scorer_preprocess()

    scorers = []
    
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)

        stats_path = scorer_path.parent.parent / f"{scorer_name}_stats.json"

        if scorer_name == "simulation":
            scorer = OpenAISimulator(llm_client, tokenizer=tokenizer, all_at_once=False)
        elif scorer_name == "fuzz":
            scorer = FuzzingScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
                stats_path=stats_path,
            )
        elif scorer_name == "detection":
            scorer = DetectionScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
                stats_path=stats_path,
            )
        else:
            raise ValueError(f"Scorer {scorer_name} not supported")

        wrapped_scorer = process_wrapper(
            scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(create_scorer_postprocess(scorer_path), score_dir=scorer_path),
        )
        scorers.append(wrapped_scorer)

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        Pipe(*scorers),
        progress_description="Scoring explanations"
    )

    if run_cfg.pipeline_num_proc > 1 and run_cfg.explainer_provider == "openrouter":
        print(
            "OpenRouter does not support multiprocessing,"
            " setting pipeline_num_proc to 1"
        )
        run_cfg.pipeline_num_proc = 1

    # Time the pipeline execution
    pipeline_start_time = time.time()
    await pipeline.run(run_cfg.pipeline_num_proc)
    pipeline_end_time = time.time()

    if not run_cfg.explainer == "none":
        if 'explainer' in locals():
            stats_path = explanations_path.parent / "explainer_stats.json"
            # explainer.stats is a defaultdict, must convert to dict for serialization.
            stats_to_save = {key: dict(value) for key, value in explainer.stats.items()}
            with open(stats_path, "wb") as f:
                f.write(orjson.dumps(stats_to_save, option=orjson.OPT_INDENT_2))


def populate_cache(
    run_cfg: RunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """
    Returns a list of hookpoints that are not already in the cache.
    For explanations, checks if any explanation files exist with the hookpoint prefix.
    For other results (like scores), checks if hookpoint directories exist.
    """
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode
    
    # Check if this is the explanations path by looking for .txt files
    # Explanations are saved as flat files like "hookpoint_latent0.txt"
    # Other results (like scores) are saved in hookpoint subdirectories
    sample_files = list(results_path.glob("*.txt"))
    is_explanations_path = len(sample_files) > 0
    
    if is_explanations_path:
        # For explanations: check if any files exist with the hookpoint prefix
        in_results_path = set()
        for file in results_path.glob("*.txt"):
            # Extract hookpoint from filename like "layers.32_latent0.txt"
            filename = file.stem  # Remove .txt
            # Split on "_latent" to get the hookpoint part
            if "_latent" in filename:
                hookpoint = filename.split("_latent")[0]
                in_results_path.add(hookpoint)
    else:
        # For scores and other results: check for hookpoint directories
        in_results_path = {x.name for x in results_path.glob("*") if x.is_dir()}
    
    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant_hookpoints = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant_hookpoints = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]
    if not non_redundant_hookpoints:
        print(f"Files found in {results_path}, skipping...")
    return non_redundant_hookpoints


def scorers_need_non_activating_examples(scorers: list[str]) -> bool:
    """
    Determine if any of the configured scorers require non-activating examples.
    
    Args:
        scorers: List of scorer names
        
    Returns:
        True if any scorer requires non-activating examples
    """
    # Scorers that always require non-activating examples
    require_non_activating = {"fuzz", "detection"}
    
    # Scorers that can use non-activating examples if available
    can_use_non_activating = {"simulation"}
    
    return any(scorer in require_non_activating for scorer in scorers)


def non_redundant_neighbour_hookpoints(
    hookpoints: list[str],
    neighbours_path: Path,
    neighbour_type: str,
    overwrite: bool,
) -> list[str]:
    """
    Returns a list of hookpoints that don't have neighbour files computed yet.
    Neighbour files are saved as {hookpoint}-{neighbour_type}.json
    """
    if overwrite:
        print("Overwriting neighbours from", neighbours_path)
        return hookpoints
    
    existing_files = [f.name for f in neighbours_path.glob("*.json")]
    non_redundant_hookpoints = []
    
    for hookpoint in hookpoints:
        expected_filename = f"{hookpoint}-{neighbour_type}.json"
        if expected_filename not in existing_files:
            non_redundant_hookpoints.append(hookpoint)
    
    if not non_redundant_hookpoints:
        print(f"Neighbour files found in {neighbours_path}, skipping...")
    
    return non_redundant_hookpoints


async def run(
    run_cfg: RunConfig,
):
    base_path = Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)

    run_cfg.save_json(base_path / "run_config.json", indent=4)

    if run_cfg.shared_cache_path:
        # Use custom shared cache directory
        shared_cache_base = Path(run_cfg.shared_cache_path)
        if not shared_cache_base.is_absolute():
            # Make relative paths relative to current working directory
            shared_cache_base = Path.cwd() / shared_cache_base
        # The latents_path points to the latents subdirectory within the shared cache
        latents_path = shared_cache_base / "latents"
        print(f"Using shared activation cache: {shared_cache_base}")
        print(f"Latents directory: {latents_path}")
    else:
        # Use experiment-specific cache directory
        latents_path = base_path / "latents"

    if run_cfg.shared_explanations_path:
        # Use custom shared explanations directory
        shared_explanations_base = Path(run_cfg.shared_explanations_path)
        if not shared_explanations_base.is_absolute():
            # Make relative paths relative to current working directory
            shared_explanations_base = Path.cwd() / shared_explanations_base
        explanations_path = shared_explanations_base
        print(f"Using shared explanations directory: {explanations_path}")
    else:
        # Use experiment-specific explanations directory
        explanations_path = base_path / "explanations"

    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"

    latent_range = torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None

    hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_cfg)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

    nrh = assert_type(
        dict,
        non_redundant_hookpoints(
            hookpoint_to_sparse_encode, latents_path, "cache" in run_cfg.overwrite
        ),
    )
    if nrh:
        populate_cache(
            run_cfg,
            model,
            nrh,
            latents_path,
            tokenizer,
            transcode,
        )

    del model, hookpoint_to_sparse_encode
    
    # Clear lingering references immediately
    gc.collect()
    torch.cuda.empty_cache()
    
    if run_cfg.constructor_cfg.non_activating_source in ["co-occurrence", "decoder_similarity", "encoder_similarity"]:
        nrh = non_redundant_neighbour_hookpoints(
            hookpoints, 
            neighbours_path, 
            run_cfg.constructor_cfg.non_activating_source,
            "neighbours" in run_cfg.overwrite
        )
        if nrh:
            create_neighbours(
                run_cfg,
                latents_path,
                neighbours_path,
                nrh,
            )
    else:
        print("Skipping neighbour creation")

    # Determine scorer model name once (fall back to explainer model)
    scorer_model_name = run_cfg.scorer_model if getattr(run_cfg, "scorer_model", None) else run_cfg.explainer_model

    # Initialize timing tracking
    explanation_time = 0.0
    scoring_time = 0.0
    total_start_time = time.time()

    nrh = assert_type(
        list,
        non_redundant_hookpoints(
            hookpoints, explanations_path, "scores" in run_cfg.overwrite
        ),
    )
    existing_llm_client = None
    if nrh:
        # Stage 1: Generate explanations and possibly return a live client to reuse
        existing_llm_client, explanation_time = await generate_explanations(
            run_cfg,
            latents_path,
            explanations_path,
            nrh,
            tokenizer,
            latent_range,
            scorer_model_name=scorer_model_name,
        )

    nrh = assert_type(
        list,
        non_redundant_hookpoints(
            hookpoints, scores_path, "scores" in run_cfg.overwrite
        ),
    )
    if nrh:
        # Stage 2: Run scoring using explanations on disk (may reuse a live client)
        scoring_time = await run_scoring(
            run_cfg,
            latents_path,
            explanations_path,
            scores_path,
            neighbours_path,
            nrh,
            tokenizer,
            latent_range,
            existing_llm_client=existing_llm_client,
        )

    # Calculate total time and save comprehensive timing data
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Save comprehensive timing data to JSON file
    timing_data = {
        "explanation_time_seconds": explanation_time,
        "scoring_time_seconds": scoring_time,
        "total_time_seconds": total_time,
        "scorers_used": run_cfg.scorers
    }
    timing_path = base_path / "timing.json"
    with open(timing_path, "wb") as f:
        f.write(orjson.dumps(timing_data, option=orjson.OPT_INDENT_2))

    if run_cfg.verbose:
        log_results(
            scores_path, visualize_path, latents_path, hookpoints, run_cfg.scorers
        )


if __name__ == "__main__":
    # Configure logging for CLI usage
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("delphi.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    parser = ArgumentParser()
    parser.add_arguments(RunConfig, dest="run_cfg")
    args = parser.parse_args()

    asyncio.run(run(args.run_cfg))
