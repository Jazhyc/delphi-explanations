from pathlib import Path
from typing import Optional

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def plot_firing_vs_f1(
    latent_df: pd.DataFrame, num_tokens: int, out_dir: Path, run_label: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for module, module_df in latent_df.groupby("module"):
        module_df = module_df.copy()
        module_df["firing_rate"] = module_df["firing_count"] / num_tokens
        fig = px.scatter(module_df, x="firing_rate", y="f1_score", log_x=True)
        fig.update_layout(
            xaxis_title="Firing rate", yaxis_title="F1 score", xaxis_range=[-5.4, 0]
        )
        fig.write_image(out_dir / f"{run_label}_{module}_firing_rates.pdf")


def import_plotly():
    """Import plotly with mitigiation for MathJax bug."""
    try:
        import plotly.express as px  # type: ignore
        import plotly.io as pio  # type: ignore
    except ImportError:
        raise ImportError(
            "Plotly is not installed.\n"
            "Please install it using `pip install plotly`, "
            "or install the `[visualize]` extra."
        )
    pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
    return px


def compute_auc(df: pd.DataFrame) -> float | None:
    if not df.probability.nunique():
        return None

    valid_df = df[df.probability.notna()]

    return roc_auc_score(valid_df.activating, valid_df.probability)  # type: ignore


def plot_accuracy_hist(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    for label in df["score_type"].unique():
        fig = px.histogram(
            df[df["score_type"] == label],
            x="accuracy",
            nbins=100,
            title=f"Accuracy distribution: {label}",
        )
        fig.write_image(out_dir / f"{label}_accuracy.pdf")


def plot_per_feature_f1_scores(df: pd.DataFrame, out_dir: Path):
    """Plots a histogram of F1 scores, one score per latent feature."""
    out_dir.mkdir(exist_ok=True, parents=True)
    # We only need one F1 score per feature, so we drop duplicates.
    unique_features_df = df.drop_duplicates(subset=["module", "latent_idx", "score_type"])

    for label in unique_features_df["score_type"].unique():
        fig = px.histogram(
            unique_features_df[unique_features_df["score_type"] == label],
            x="f1_score",
            title=f"Per-Feature F1 Score Distribution: {label}",
        )
        fig.write_image(out_dir / f"{label}_f1_distribution.pdf")


def plot_model_comparison_density(model_results: dict, out_dir: Path):
    """Plot density vs accuracy for different models."""
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Combine all model data
    all_data = []
    for model_name, data in model_results.items():
        for score_type in data['processed_df']['score_type'].unique():
            score_data = data['processed_df'][data['processed_df']['score_type'] == score_type].iloc[0]
            all_data.append({
                'model': model_name,
                'score_type': score_type,
                'accuracy': score_data['accuracy'],
                'f1_score': score_data['f1_score'],
                'precision': score_data['precision'],
                'recall': score_data['recall']
            })
    
    df = pd.DataFrame(all_data)
    
    # Create density plots for each score type
    for score_type in df['score_type'].unique():
        score_df = df[df['score_type'] == score_type]
        
        fig = px.violin(
            score_df, 
            x='model', 
            y='accuracy',
            title=f'Accuracy Distribution by Model - {score_type.title()}',
            points="all"
        )
        fig.update_layout(
            yaxis_range=[0, 1],
            xaxis_title="Model",
            yaxis_title="Accuracy",
            xaxis={'tickangle': 45}
        )
        fig.write_image(out_dir / f"accuracy_density_{score_type}.pdf")


def plot_model_comparison_bar(model_results: dict, out_dir: Path):
    """Plot bar chart of mean accuracy for each model."""
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Combine all model data
    all_data = []
    for model_name, data in model_results.items():
        for score_type in data['processed_df']['score_type'].unique():
            score_data = data['processed_df'][data['processed_df']['score_type'] == score_type].iloc[0]
            all_data.append({
                'model': model_name,
                'score_type': score_type,
                'accuracy': score_data['accuracy'],
                'f1_score': score_data['f1_score'],
                'precision': score_data['precision'],
                'recall': score_data['recall']
            })
    
    df = pd.DataFrame(all_data)
    
    # Create bar plots for each score type
    for score_type in df['score_type'].unique():
        score_df = df[df['score_type'] == score_type]
        
        fig = px.bar(
            score_df, 
            x='model', 
            y='accuracy',
            title=f'Mean Accuracy by Model - {score_type.title()}',
            text='accuracy'
        )
        fig.update_layout(
            yaxis_range=[0, 1],
            xaxis_title="Model",
            yaxis_title="Accuracy",
            xaxis={'tickangle': 45}
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.write_image(out_dir / f"accuracy_bar_{score_type}.pdf")


def plot_token_usage_analysis(model_stats: dict, out_dir: Path):
    """Plot token usage statistics for different models."""
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for token analysis
    token_data = []
    for model_name, stats in model_stats.items():
        if stats:  # Check if stats exist
            for explainer_type, explainer_stats in stats.items():
                token_data.append({
                    'model': model_name,
                    'explainer_type': explainer_type,
                    'prompt_tokens': explainer_stats.get('prompt_tokens', 0),
                    'completion_tokens': explainer_stats.get('completion_tokens', 0),
                    'total_time': explainer_stats.get('total_time', 0),
                    'count': explainer_stats.get('count', 0)
                })
    
    if not token_data:
        print("No token usage data available")
        return
    
    df = pd.DataFrame(token_data)
    df['total_tokens'] = df['prompt_tokens'] + df['completion_tokens']
    df['avg_tokens_per_call'] = df['total_tokens'] / df['count'].replace(0, 1)
    df['avg_time_per_call'] = df['total_time'] / df['count'].replace(0, 1)
    
    # Plot total tokens used
    fig = px.bar(
        df, 
        x='model', 
        y='total_tokens',
        color='explainer_type',
        title='Total Tokens Used by Model',
        text='total_tokens'
    )
    fig.update_layout(xaxis={'tickangle': 45})
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.write_image(out_dir / "total_tokens_usage.pdf")
    
    # Plot average tokens per call
    fig = px.bar(
        df, 
        x='model', 
        y='avg_tokens_per_call',
        color='explainer_type',
        title='Average Tokens per Explanation Call',
        text='avg_tokens_per_call'
    )
    fig.update_layout(xaxis={'tickangle': 45})
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.write_image(out_dir / "avg_tokens_per_call.pdf")
    fig.write_html(out_dir / "avg_tokens_per_call.html")
    
    # Plot average time per call
    fig = px.bar(
        df, 
        x='model', 
        y='avg_time_per_call',
        color='explainer_type',
        title='Average Time per Explanation Call (seconds)',
        text='avg_time_per_call'
    )
    fig.update_layout(xaxis={'tickangle': 45})
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.write_image(out_dir / "avg_time_per_call.pdf")
    fig.write_html(out_dir / "avg_time_per_call.html")


def plot_roc_curve(df: pd.DataFrame, out_dir: Path):
    if not df.probability.nunique():
        return

    # filter out NANs
    valid_df = df[df.probability.notna()]

    fpr, tpr, _ = roc_curve(valid_df.activating, valid_df.probability)
    auc = roc_auc_score(valid_df.activating, valid_df.probability)
    fig = go.Figure(
        data=[
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")),
        ]
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="FPR",
        yaxis_title="TPR",
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.write_image(out_dir / "roc_curve.pdf")


def compute_confusion(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    df_valid = df[df["prediction"].notna()]
    act = df_valid["activating"].astype(bool)

    total = len(df_valid)
    pos = act.sum()
    neg = total - pos

    tp = ((df_valid.prediction >= threshold) & act).sum()
    tn = ((df_valid.prediction < threshold) & ~act).sum()
    fp = ((df_valid.prediction >= threshold) & ~act).sum()
    fn = ((df_valid.prediction < threshold) & act).sum()

    assert fp <= neg and tn <= neg and tp <= pos and fn <= pos

    return dict(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_examples=total,
        total_positives=pos,
        total_negatives=neg,
        failed_count=len(df_valid) - total,
    )


def compute_classification_metrics(conf: dict) -> dict:
    tp = conf["true_positives"]
    tn = conf["true_negatives"]
    fp = conf["false_positives"]
    fn = conf["false_negatives"]
    total = conf["total_examples"]
    pos = conf["total_positives"]
    neg = conf["total_negatives"]

    assert pos + neg == total, "pos + neg must equal total"

    # accuracy = (tp + tn) / total if total > 0 else 0
    balanced_accuracy = (
        (tp / pos if pos > 0 else 0) + (tn / neg if neg > 0 else 0)
    ) / 2

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / pos if pos > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return dict(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=balanced_accuracy,
        true_positive_rate=tp / pos if pos > 0 else 0,
        true_negative_rate=tn / neg if neg > 0 else 0,
        false_positive_rate=fp / neg if neg > 0 else 0,
        false_negative_rate=fn / pos if pos > 0 else 0,
        total_examples=total,
        total_positives=pos,
        total_negatives=neg,
        positive_class_ratio=pos / total if total > 0 else 0,
        negative_class_ratio=neg / total if total > 0 else 0,
    )


def load_data(scores_path: Path, latents_path: Path, modules: list[str]):
    """Load all on-disk data into a single DataFrame."""

    def parse_score_file(path: Path) -> pd.DataFrame:
        """
        Load a score file and return a raw DataFrame
        """
        try:
            data = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            print(f"Error decoding JSON from {path}. Skipping file.")
            return pd.DataFrame()

        latent_idx = int(path.stem.split("latent")[-1])

        return pd.DataFrame(
            [
                {
                    "text": "".join(ex["str_tokens"]),
                    "distance": ex["distance"],
                    "activating": ex["activating"],
                    "prediction": ex["prediction"],
                    "probability": ex["probability"],
                    "correct": ex["correct"],
                    "activations": ex["activations"],
                    "latent_idx": latent_idx,
                }
                for ex in data
            ]
        )

    counts_file = latents_path.parent / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(counts_file, weights_only=True) if counts_file.exists() else {}
    if not all(module in counts for module in modules):
        print("Missing firing counts for some modules, setting counts to None.")
        print(f"Missing modules: {[m for m in modules if m not in counts]}")
        counts = None

    # Collect per-latent data
    latent_dfs = []
    for score_type_dir in scores_path.iterdir():
        if not score_type_dir.is_dir():
            continue
        for module in modules:
            for file in score_type_dir.glob(f"*{module}*"):
                latent_idx = int(file.stem.split("latent")[-1])

                latent_df = parse_score_file(file)
                latent_df["score_type"] = score_type_dir.name
                latent_df["module"] = module
                latent_df["latent_idx"] = latent_idx
                if counts:
                    latent_df["firing_count"] = (
                        counts[module][latent_idx].item()
                        if latent_idx in counts[module]
                        else None
                    )

                latent_dfs.append(latent_df)

    return pd.concat(latent_dfs, ignore_index=True), counts


def frequency_weighted_f1(
    df: pd.DataFrame, counts: dict[str, torch.Tensor]
) -> float | None:
    rows = []
    for (module, latent_idx), grp in df.groupby(["module", "latent_idx"]):
        f1 = compute_classification_metrics(compute_confusion(grp))["f1_score"]
        fire = counts[module][latent_idx].item()
        rows.append(
            {
                "module": module,
                "latent_idx": latent_idx,
                "f1_score": f1,
                "firing_count": fire,
            }
        )

    latent_df = pd.DataFrame(rows)

    per_module_f1 = []
    for module in latent_df["module"].unique():
        module_df = latent_df[latent_df["module"] == module]

        firing_weights = counts[module][module_df["latent_idx"]].float()
        total_weight = firing_weights.sum()
        if total_weight == 0:
            continue

        f1_tensor = torch.as_tensor(module_df["f1_score"].values, dtype=torch.float32)
        module_f1 = (f1_tensor * firing_weights).sum() / firing_weights.sum()
        per_module_f1.append(module_f1)

    overall_frequency_weighted_f1 = torch.stack(per_module_f1).mean()
    return (
        overall_frequency_weighted_f1.item()
        if not overall_frequency_weighted_f1.isnan()
        else None
    )


def get_agg_metrics(
    latent_df: pd.DataFrame, counts: Optional[dict[str, torch.Tensor]]
) -> pd.DataFrame:
    processed_rows = []
    for score_type, group_df in latent_df.groupby("score_type"):
        conf = compute_confusion(group_df)
        class_m = compute_classification_metrics(conf)
        auc = compute_auc(group_df)
        f1_w = frequency_weighted_f1(group_df, counts) if counts else None

        row = {
            "score_type": score_type,
            **conf,
            **class_m,
            "auc": auc,
            "weighted_f1": f1_w,
        }
        processed_rows.append(row)

    return pd.DataFrame(processed_rows)


def add_latent_f1(latent_df: pd.DataFrame) -> pd.DataFrame:
    f1s = (
        latent_df.groupby(["module", "latent_idx", "score_type"])
        .apply(
            lambda g: compute_classification_metrics(compute_confusion(g))["f1_score"]
        )
        .reset_index(name="f1_score")  # <- naive (un-weighted) F1
    )
    return latent_df.merge(f1s, on=["module", "latent_idx", "score_type"])


def log_results(
    scores_path: Path,
    viz_path: Path,
    latents_path: Path,
    modules: list[str],
    scorer_names: list[str],
):
    import_plotly()

    latent_df, counts = load_data(scores_path, latents_path, modules)
    latent_df = latent_df[latent_df["score_type"].isin(scorer_names)]
    latent_df = add_latent_f1(latent_df)

    plot_firing_vs_f1(
        latent_df, num_tokens=10_000_000, out_dir=viz_path, run_label=scores_path.name
    )

    if latent_df.empty:
        print("No data found")
        return

    dead = sum((counts[m] == 0).sum().item() for m in modules)
    print(f"Number of dead features: {dead}")
    print(f"Number of interpreted live features: {len(latent_df)}")

    # Load constructor config for run
    with open(scores_path.parent / "run_config.json", "r") as f:
        run_cfg = orjson.loads(f.read())
    constructor_cfg = run_cfg.get("constructor_cfg", {})
    min_examples = constructor_cfg.get("min_examples", None)
    print("min examples", min_examples)

    if min_examples is not None:
        uninterpretable_features = sum(
            [(counts[m] < min_examples).sum() for m in modules]
        )
        print(
            f"Number of features below the interpretation firing"
            f" count threshold: {uninterpretable_features}"
        )

    plot_roc_curve(latent_df, viz_path)

    processed_df = get_agg_metrics(latent_df, counts)

    plot_per_feature_f1_scores(latent_df, viz_path)
    for score_type in processed_df.score_type.unique():
        score_type_summary = processed_df[processed_df.score_type == score_type].iloc[0]
        print(f"\n--- {score_type.title()} Metrics ---")
        print(f"Class-Balanced Accuracy: {score_type_summary['accuracy']:.3f}")
        print(f"F1 Score: {score_type_summary['f1_score']:.3f}")
        print(f"Frequency-Weighted F1 Score: {score_type_summary['weighted_f1']:.3f}")
        print(
            "Note: the frequency-weighted F1 score is computed over each"
            " hookpoint and averaged"
        )
        print(f"Precision: {score_type_summary['precision']:.3f}")
        print(f"Recall: {score_type_summary['recall']:.3f}")
        # Only print AUC if unbalanced AUC is not -1.
        if score_type_summary["auc"] is not None:
            print(f"AUC: {score_type_summary['auc']:.3f}")
        else:
            print("Logits not available.")

        fractions_failed = [
            score_type_summary["failed_count"]
            / (
                (
                    score_type_summary["total_examples"]
                    + score_type_summary["failed_count"]
                )
            )
        ]
        print(
            f"""Average fraction of failed examples: \
{sum(fractions_failed) / len(fractions_failed)}"""
        )

        print("\nConfusion Matrix:")
        print(
            f"True Positive Rate:  {score_type_summary['true_positive_rate']:.3f} "
            f"({score_type_summary['true_positives'].sum()})"
        )
        print(
            f"True Negative Rate:  {score_type_summary['true_negative_rate']:.3f} "
            f"({score_type_summary['true_negatives'].sum()})"
        )
        print(
            f"False Positive Rate: {score_type_summary['false_positive_rate']:.3f} "
            f"({score_type_summary['false_positives'].sum()})"
        )
        print(
            f"False Negative Rate: {score_type_summary['false_negative_rate']:.3f} "
            f"({score_type_summary['false_negatives'].sum()})"
        )

        print("\nClass Distribution:")
        print(f"""Positives: {score_type_summary['total_positives'].sum():.0f}""")
        print(f"""Negatives: {score_type_summary['total_negatives'].sum():.0f}""")
        print(f"Total: {score_type_summary['total_examples'].sum():.0f}")
