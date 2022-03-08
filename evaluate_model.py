"""
Evaluate a given model on a given dataset.
"""
import click

from comet_ml import Experiment

from config import DEFAULT_OUT_DIR
from dataset import get_dataset_by_name, AVAIL_DATASETS, MLMA_RAW_DATASETS
import metrics
import pandas as pd
import logging
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import glob

logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--dataset", type=click.Choice(AVAIL_DATASETS), required=True)
@click.option("--model_path", type=str)
@click.option(
    "--subgroups_path", type=str, help="Path to the subgroups file.", default=None
)
@click.option(
    "--n_jobs",
    type=int,
    help="Used to parallelize the evaluation of bias metrics",
    default=4,
)
@click.option("--cpu_only", is_flag=True)
@click.option("--no_bias_metrics", is_flag=True)
@click.option("--model_suffix", type=str)
@click.option("--out_folder", type=str, default=DEFAULT_OUT_DIR)
@click.option("--log_comet", is_flag=True)
@click.option("--ckpt_pattern", type=str, default=None)
@click.option("--src_tokenizer", type=str, default=None)
@click.option("--src_model", type=str, default=None)
def evaluate(
    dataset,
    model_path,
    subgroups_path,
    n_jobs,
    cpu_only,
    no_bias_metrics,
    model_suffix,
    out_folder,
    log_comet,
    ckpt_pattern,
    src_tokenizer,
    src_model,
):
    os.makedirs(out_folder, exist_ok=True)

    hparams = locals()

    if src_model:
        logger.info(f"Using model {src_model}")
        model_name = src_model.split("/")[1]
        model_path = src_model
    else:
        model_name = os.path.basename(model_path)

    # lm_ws is created by Kennedy's simulation but shouldn't be used
    if model_name.startswith("lm_ws"):
        logger.info(f"Skipping the model {model_name}...")
        return

    if model_suffix:
        model_name = f"{model_name}-{model_suffix}"

    hparams["model"] = model_name

    if log_comet:
        experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name="unbias-text-classifiers",
            log_code=False,
            log_graph=False,
        )
        experiment.set_name(f"evaluate_{dataset}")
        experiment.log_parameters(hparams)
        experiment.add_tag("evaluation")

    logger.info(f"BEGIN evaluating {model_name} on {dataset}")

    #  Get dataset splits. Discard train and dev
    _, _, test = get_dataset_by_name(dataset)
    if log_comet:
        experiment.log_other("test_size", len(test))
    y_true = test.get_labels()

    scores_file = os.path.join(out_folder, f"scores_{model_name}_{dataset}.pt")
    if os.path.exists(os.path.join(out_folder, scores_file)):
        logger.info(
            "Scores already exist. Loading them and continuing the evaluation..."
        )
        scores = torch.load(os.path.join(out_folder, scores_file))
    else:
        scores = evaluate_bert(test, model_path, cpu_only, ckpt_pattern, src_tokenizer)

    #  Compute classification metrics based on scores
    logger.info("Evaluating standard performance metrics...")
    perf, y_pred = metrics.evaluate_metrics(y_true=y_true, y_score=scores, th=0.5)
    if log_comet:
        experiment.log_metrics(perf)

    # Save scores and classification metrics locally and on Comet
    torch.save(scores, scores_file)
    pd.Series(perf).to_frame().to_csv(
        os.path.join(out_folder, f"class_metrics_{model_name}_{dataset}.csv")
    )
    if log_comet:
        experiment.log_asset(scores_file)
        experiment.log_metrics(perf)
        experiment.log_confusion_matrix(
            y_true=y_true, y_predicted=y_pred.astype(int).tolist()
        )

    # run the evaluation on MLMA
    if dataset in MLMA_RAW_DATASETS:
        logger.info("Processing MLMA per-target performance")
        mlma_results = compute_metrics_on_mlma(test, y_true, scores)

        mlma_df = pd.DataFrame(
            [r[3] for r in mlma_results], index=[r[0] for r in mlma_results]
        )
        mlma_df.to_csv(
            os.path.join(
                out_folder, f"class_metrics_by_target_{model_name}_{dataset}.csv"
            )
        )

    if no_bias_metrics:
        if log_comet:
            experiment.add_tag("no_bias_metrics")
        logger.info(f"END {model_name} (skipped bias metrics)")
        return

    # --- Evaluation of bias metrics ---

    #  Read subgroups and add a dummy column indicating its presence
    with open(subgroups_path) as fp:
        subgroups = [line.strip().split("\t")[0] for line in fp.readlines()]

    logging.info(f"Found subgroups: {subgroups}")
    if log_comet:
        experiment.log_other("subgroups", subgroups)
        experiment.log_other("subgroups_count", len(subgroups))

    #  this df is required by the Jigsaw's code for bias metrics
    data_df = pd.DataFrame(
        {"text": test.get_texts(), "label": y_true, model_name: scores}
    )
    data_df = metrics.add_subgroup_columns_from_text(data_df, "text", subgroups)

    logger.info("Evaluating bias metrics (parallel)...")
    bias_records = Parallel(n_jobs=n_jobs)(
        delayed(metrics.compute_bias_metrics_for_subgroup_and_model)(
            dataset=data_df,
            subgroup=subg,
            model=model_name,
            label_col="label",
            include_asegs=True,
        )
        for subg in tqdm(subgroups)
    )

    bias_terms_file = os.path.join(out_folder, f"bias_terms_{model_name}_{dataset}.csv")
    per_term_df = pd.DataFrame(bias_records)
    per_term_df.to_csv(bias_terms_file, index=False)
    if log_comet:
        experiment.log_table(bias_terms_file)

    # Average bias metrics across subgroups
    records_df = per_term_df.drop(columns=["test_size", "subgroup"])

    # TODO: ignore nans?
    #  compute the mean value of each bias metric across subgroups. Here we use
    #  1. power mean (Jigsaw's Kaggle competition). It weights more subgroups where the metric is low.
    #  2. arithmetic mean
    power_mean_values = metrics.power_mean(records_df.values, -5, ignore_nans=True)
    mean_values = metrics.power_mean(records_df.values, 1, ignore_nans=True)

    power_mean_dict = {
        f"{name}_power_mean": v
        for name, v in zip(records_df.columns, power_mean_values)
    }
    mean_dict = {f"{name}_mean": v for name, v in zip(records_df.columns, mean_values)}

    # The final summary metric is the average between:
    # overall AUC, subgroup_auc, bpsn_auc, bnsp_auc
    summary_metric_pm = np.nanmean(
        np.array(
            [
                perf["AUC"],
                power_mean_dict["subgroup_auc_power_mean"],
                power_mean_dict["bpsn_auc_power_mean"],
                power_mean_dict["bnsp_auc_power_mean"],
            ]
        )
    )

    summary_metric = np.nanmean(
        np.array(
            [
                perf["AUC"],
                mean_dict["subgroup_auc_mean"],
                mean_dict["bpsn_auc_mean"],
                mean_dict["bnsp_auc_mean"],
            ]
        )
    )

    bias_metrics = {
        **power_mean_dict,
        **mean_dict,
        "summary_power_mean": summary_metric_pm,
        "summary_mean": summary_metric,
    }

    #  Add False Positive and False Negative Equality Difference (Equality of Odds)
    bias_metrics["fped"] = per_term_df[metrics.FPR_GAP].abs().sum()
    bias_metrics["fped_mean"] = per_term_df[metrics.FPR_GAP].abs().mean()
    bias_metrics["fped_std"] = per_term_df[metrics.FPR_GAP].abs().std()
    bias_metrics["fned"] = per_term_df[metrics.FNR_GAP].abs().sum()
    bias_metrics["fned_mean"] = per_term_df[metrics.FNR_GAP].abs().mean()
    bias_metrics["fned_std"] = per_term_df[metrics.FNR_GAP].abs().std()

    if log_comet:
        experiment.log_metrics(bias_metrics)
    pd.Series(bias_metrics).to_frame().to_csv(
        os.path.join(out_folder, f"bias_metrics_{model_name}_{dataset}.csv")
    )
    logger.info(f"END {model_name}")


def evaluate_bert(
    dataset,
    model_dir,
    cpu_only: bool,
    ckpt_pattern,
    src_tokenizer,
    batch_size=64,
    max_sequence_length=120,
):
    """Run evaluation on Kennedy's BERT."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader

    device = "cuda:0" if (torch.cuda.is_available() and not cpu_only) else "cpu"
    logger.info(f"Device: {device}")

    if ckpt_pattern:
        from train_bert import LMForSequenceClassification

        ckpt_path = glob.glob(os.path.join(model_dir, f"*{ckpt_pattern}*"))[0]
        logger.info(f"Loading ckpt {ckpt_path}")
        model = LMForSequenceClassification.load_from_checkpoint(ckpt_path).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    model.eval()

    if src_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(src_tokenizer)
    else:
        logger.info(f"Src tokenizer not specified, using {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    probs = list()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            encodings = tokenizer(
                batch["text"],
                add_special_tokens=True,  #  they use BERT's special tokens
                padding=True,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            ).to(device)

            output = model(**encodings)
            batch_probs = output["logits"].softmax(-1)  # batch_size x 2
            probs.append(batch_probs)

    probs = torch.cat(probs, dim=0)

    #  return probabilities for the positive label only
    return probs[:, 1].cpu()


def compute_metrics_on_mlma(mlma_data, y_true, scores):
    targets = mlma_data.data.target.unique()
    logger.info(f"Targets found {targets}")

    target_mask = pd.get_dummies(mlma_data.data["target"]).astype(bool)

    # y_true is a list, y_pred a np.array, scores a torch.tensor
    y_true = np.array(y_true)

    results = list()
    for target in targets:
        mask = target_mask[target].values
        perf, y_pred = metrics.evaluate_metrics(
            y_true=y_true[mask], y_score=scores[mask], th=0.5
        )
        perf["size"] = y_true[mask].size
        results.append((target, y_true[mask], scores[mask], perf))
    return results


if __name__ == "__main__":
    evaluate()
