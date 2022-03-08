from os.path import exists, join
import pandas as pd
import torch
import logging

from transformers import AutoModelForSequenceClassification
from train_bert import compute_negative_entropy, LMForSequenceClassification
from dataset import get_dataset_by_name, TokenizerDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
import glob
import numpy as np
from IPython.display import display
import os
from os.path import join
import re
import torch
from collections import namedtuple
import pdb


logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ScoreHandler:
    """Standardize how scores are saved and loaded for a given model & dataset."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def save_scores(self, scores, root_dir: str, column_name: str, dataset: str):
        """Save the scores for a model on a dataset.

        It uses a single csv file per dataset. Each column refers to the scores of a
        single dataset.

        Return: (datafram with scores, epath of the file containing the scores)
        """
        file_name = f"scores_{dataset}.csv"
        file_path = join(root_dir, file_name)
        df = pd.read_csv(file_path) if exists(file_path) else self.dataset.data.copy()

        if column_name in df.columns:
            logging.info(f"Scores for {column_name} are present. Overriding them...")
        df[column_name] = scores
        df.to_csv(file_path, index=False)

        return df, file_path


def load_model_from_folder(model_dir, pattern=None):
    if pattern:
        ckpt = glob.glob(join(model_dir, f"*{pattern}*"))[0]
    else:
        ckpt = glob.glob(f"{model_dir}/*.ckpt")[0]

    print("Loading", ckpt)

    if pattern:
        model = LMForSequenceClassification.load_from_checkpoint(ckpt)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model


def join_subwords(tokens):
    span_start_idx = -1
    spans = list()
    for i, t in enumerate(tokens):
        if t.startswith("#") and span_start_idx == -1:
            span_start_idx = i - 1
            continue
        if not t.startswith("#") and span_start_idx != -1:
            spans.append((span_start_idx, i))
            span_start_idx = -1

    #  span open at the end
    if span_start_idx != -1:
        spans.append((span_start_idx, len(tokens)))

    merged_tkns = list()
    pop_idxs = list()
    for span in spans:
        merged = "".join([t.strip("#") for t in tokens[span[0] : span[1]]])
        merged_tkns.append(merged)

        #  indexes to remove in the final sequence
        for pop_idx in range(span[0] + 1, span[1]):
            pop_idxs.append(pop_idx)

    new_tokens = tokens.copy()
    for i, (span, merged) in enumerate(zip(spans, merged_tkns)):
        new_tokens[span[0]] = merged  #  substitue with whole word

    mask = np.ones(len(tokens))
    mask[pop_idxs] = 0
    new_tokens = np.array(new_tokens)[mask == 1]

    assert len(new_tokens) == len(tokens) - len(pop_idxs)
    return new_tokens, pop_idxs, spans


def average_2d_over_spans(tensor, spans, reduce_fn="mean"):
    #  print("Spans #", spans)
    slices = list()

    last_span = None
    for span in spans:

        # first slice
        if last_span is None:
            slices.append(tensor[:, : span[0]])
        else:
            slices.append(tensor[:, last_span[1] : span[0]])

        # average over the subwords
        if reduce_fn == "mean":
            slices.append(tensor[:, span[0] : span[1]].mean(-1).unsqueeze(-1))
        else:
            slices.append(tensor[:, span[0] : span[1]].sum(-1).unsqueeze(-1))

        last_span = span

    #  last slice
    if spans[-1][1] != tensor.shape[1]:
        slices.append(tensor[:, last_span[1] :])

    res = torch.cat(slices, dim=1)
    #  print("After average:", res.shape)
    return res


def get_scores(y_true, scores_path):
    scores = torch.load(scores_path)
    y_pred = torch.zeros(scores.shape[0]).masked_fill(scores >= 0.5, 1)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fp = torch.zeros(scores.shape[0]).masked_fill(fp_mask, 1)
    fp_indexes = torch.nonzero(fp).squeeze(-1)

    print(f"Found {fp_indexes.shape[0]} FPs")
    return {"scores": scores, "y_pred": y_pred, "fp_indexes": fp_indexes}


#### VISUALIZATION: ENTROPY ####


def show_entropy(
    models,
    tokenizer,
    max_sequence_length,
    data,
    names,
    n_samples=2,
    idxs=None,
    regularization="entropy",
    join=False,
    layers_mean=False,
    prompt=None,
    exp=False,
    remove_special=False,
    labelsize=15,
    titlesize=15,
    set_figsize=True,
    set_tightlayout=True,
):
    def process_text(idx, text):
        with torch.no_grad():
            print(text)
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )

            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

            if remove_special:
                tokens = tokens[1:-1]

            #  print("Len:", len(tokens), "tokens:", tokens)

            if join:
                # join subwords for better visualization
                new_tokens, pop_idxs, spans = join_subwords(tokens)
                #  print("Len new tokens", len(new_tokens))
                tokens = new_tokens

            heatmap_list = list()
            final_entropies = list()
            y_scores = list()
            for i, (model, name) in enumerate(zip(models, names)):
                if regularization == "entropy":
                    output = model(**encoding, output_attentions=True)
                    reg_target = output["attentions"]
                else:
                    output = model(**encoding, output_norms=True)
                    norms = output["norms"]
                    afx_norms = [t[1] for t in norms]
                    reg_target = afx_norms

                logits = output["logits"]
                y_score = logits.softmax(-1)[0, 1]
                print(y_score)

                neg_entropy, entropies = compute_negative_entropy(
                    reg_target, encoding["attention_mask"], return_values=True
                )
                #  print("Entropies shape:", entropies[0].shape)

                #  join_subwords(entropies, tokens)
                #  print(name, "Final entropy: ", -neg_entropy.item())
                entropies = -entropies[0]  # take positive entropy
                entropies = torch.flipud(entropies)  #  top layers are placed to the top

                # average subwords
                if join and len(spans) > 0:
                    entropies = average_2d_over_spans(entropies, spans)

                if layers_mean:
                    entropies = entropies.mean(0).unsqueeze(0)

                if exp:
                    entropies = (1 / entropies).log()

                if remove_special:
                    entropies = entropies[:, 1:-1]

                heatmap_list.append(entropies)
                final_entropies.append(-neg_entropy.item())
                y_scores.append(y_score)

            #### VISUALIZATION ####

            if layers_mean:
                figsize = (12, 2 * len(models))
            else:
                figsize = (6 * len(models), 6)

            if set_figsize:
                fig = plt.figure(constrained_layout=False, figsize=figsize)
            else:
                fig = plt.figure(constrained_layout=False)

            if regularization == "entropy":
                fig.suptitle(
                    f"H: Entropy on Attention (a), ID:{idx}"
                )  # , {data[idx]}")
            else:
                fig.suptitle(
                    f"Entropy on Norm (||a*f(zx)||), ID:{idx}"
                )  # , {data[idx]}")

            if set_tightlayout:
                fig.tight_layout()

            # compute global min and global max
            heatmap_tensor = torch.stack(heatmap_list)
            glob_min = heatmap_tensor.min().item()
            glob_max = heatmap_tensor.max().item()
            #  print("Glob max:", glob_max, "Glob min", glob_min)

            for i, name in enumerate(names):
                if layers_mean:
                    gspec = fig.add_gridspec(
                        len(models), 2, width_ratios=[20, 1], wspace=0.1, hspace=0.1
                    )
                    splot = fig.add_subplot(gspec[i, 0])

                    if i == (len(names) - 1):
                        cbar_ax = fig.add_subplot(gspec[:, 1])
                        sns.heatmap(
                            heatmap_list[i],
                            ax=splot,
                            cbar=True,
                            cbar_ax=cbar_ax,
                            square=True,
                            vmin=glob_min,
                            vmax=glob_max,
                        )
                        splot.set_xticks(np.arange(heatmap_list[i].shape[-1]) + 0.5)
                        splot.set_xticklabels(tokens, rotation=90, fontsize=labelsize)
                        [t.set_fontsize(labelsize) for t in cbar_ax.get_yticklabels()]

                        # title to colorbar
                        cbar_ax.set_title(
                            "log(1/H)", fontsize=titlesize
                        ) if exp else cbar_ax.set_title("H", fontsize=titlesize)

                    else:
                        sns.heatmap(
                            heatmap_list[i],
                            ax=splot,
                            cbar=False,
                            square=True,
                            vmin=glob_min,
                            vmax=glob_max,
                        )
                        splot.set_xticklabels([])

                    splot.set_yticklabels([])
                    splot.set_title(
                        f"{name}, p(1|x)={y_scores[i]:.3f}, H={final_entropies[i]:.3f}",
                        fontsize=titlesize,
                    )

                else:
                    width_ratios = [10] * len(models)
                    width_ratios += [1]
                    gspec = fig.add_gridspec(
                        1, len(models) + 1, width_ratios=width_ratios, wspace=0.2
                    )
                    splot = fig.add_subplot(gspec[0, i])

                    if i == (len(names) - 1):
                        cbar_ax = fig.add_subplot(gspec[0, -1])
                        sns.heatmap(
                            heatmap_list[i],
                            ax=splot,
                            cbar=True,
                            cbar_ax=cbar_ax,
                            square=True,
                            vmin=glob_min,
                            vmax=glob_max,
                        )
                        [t.set_fontsize(labelsize) for t in cbar_ax.get_yticklabels()]

                        # title to colorbar
                        cbar_ax.set_title(
                            "log(1/H)", fontsize=titlesize
                        ) if exp else cbar_ax.set_title("H", fontsize=titlesize)
                    else:
                        sns.heatmap(heatmap_list[i], ax=splot, cbar=False, square=True)

                    if i == 0:
                        splot.set_ylabel("Layer", fontsize=labelsize)
                        splot.set_yticklabels(np.arange(11, -1, -1), fontsize=labelsize)
                    else:
                        splot.set_yticklabels([])

                    splot.set_xticks(np.arange(heatmap_list[i].shape[-1]) + 0.5)
                    splot.set_xticklabels(tokens, rotation=90, fontsize=labelsize)
                    splot.set_title(
                        f"{name}, p(1|x)={y_scores[i]:.3f}, H={final_entropies[i]:.3f}",
                        fontsize=titlesize,
                    )

                    #  print(len(tokens), len(axes[i].get_xticklabels()))
                    #  print(entropies.shape)
                    #  axes[i].set_xticks(np.arange(heatmap_list[i].shape[-1]))
                    # axes[i].set_xticklabels(tokens, rotation=90)
                    # axes[i].set_title(f"{name}, p(1|x)={y_scores[i]:.3f}, e={final_entropies[i]:.3f}")
                    # axes[i].set_yticklabels([])
            return fig

    if prompt:
        idx = "custom"
        text = prompt
        print("ID: ", idx, text)
        return process_text(idx, text)

    if idxs is None:
        #  pick random samples to show
        idxs = np.random.randint(len(data), size=n_samples)

    print(idxs)
    for idx in idxs:
        print("ID: ", idx, data[idx])
        process_text(idx, data[idx]["text"])


def compare_sentences(
    model,
    tokenizer,
    sentences,
    max_sequence_length=120,
    remove_special=True,
    join=True,
    show_log=True,
    labelsize=15,
    titlesize=15,
    figsize=(12, 12),
):
    processed = list()

    with torch.no_grad():

        for text in sentences:

            encoding = tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )

            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

            if remove_special:
                tokens = tokens[1:-1]

            if join:
                # join subwords for better visualization
                new_tokens, pop_idxs, spans = join_subwords(tokens)
                #  print("Len new tokens", len(new_tokens))
                tokens = new_tokens

            output = model(**encoding, output_attentions=True)
            logits = output["logits"]
            y_score = logits.softmax(-1)[0, 1]

            neg_entropy, entropies = compute_negative_entropy(
                output["attentions"], encoding["attention_mask"], return_values=True
            )
            #  print("Entropies shape:", entropies[0].shape)

            #  print(name, "Final entropy: ", -neg_entropy.item())
            entropies = -entropies[0]  # take positive entropy

            # average subwords
            if join and len(spans) > 0:
                entropies = average_2d_over_spans(entropies, spans)

            entropies = entropies.mean(0).unsqueeze(0)

            if show_log:
                entropies = (1 / entropies).log()

            if remove_special:
                entropies = entropies[:, 1:-1]

            processed.append((tokens, y_score, entropies))

    # print(processed)
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gspec = fig.add_gridspec(len(sentences) * 2, 1, hspace=2, wspace=5)

    vmin = torch.stack([p[2] for p in processed]).min().item()
    vmax = torch.stack([p[2] for p in processed]).max().item()
    print(vmin, vmax)

    for i, (tokens, y_score, entropies) in enumerate(processed):
        splot = fig.add_subplot(gspec[i, 0])

        #  cbar_ax = fig.add_subplot(gspec[:, 1])
        sns.heatmap(
            entropies,
            ax=splot,
            cbar=False,
            # cbar_ax=cbar_ax,
            square=True,
            # cmap="Reds",
            annot=False,
            vmin=vmin,
            vmax=vmax,
        )
        splot.set_xticks(np.arange(entropies.shape[-1]) + 0.5)
        splot.set_xticklabels(tokens, rotation=90, fontsize=labelsize)
        splot.set_yticklabels([])
        splot.set_title(
            f"p(1|x)={y_score:.3f}",
            fontsize=titlesize,
        )
        # [t.set_fontsize(labelsize) for t in cbar_ax.get_yticklabels()]

        # title to colorbar
        # cbar_ax.set_title(
        #     "log(1/H)", fontsize=titlesize
        # ) if exp else cbar_ax.set_title("H", fontsize=titlesize)
    # fig.tight_layout()


#### BIAS_ANALYSIS: parsing results and bias analysis


def match_pattern_concat(main_dir, pattern, verbose=True):
    """Find all files that match a patter in main_dir. Then concatenate their content into a pandas df."""
    versions = glob.glob(join(main_dir, pattern))
    if verbose:
        print(f"Found {len(versions)} versions")

    res = list()
    for version in versions:
        df = pd.read_csv(version)
        filename = os.path.basename(version)
        seed = re.search(r"([0-9]{1})", filename).group(1)
        # print(filename, seed)
        df["seed"] = seed
        res.append(df)

    return pd.concat(res)


def mean_std_across_subgroups(data: pd.DataFrame, metrics):
    print("Found the following models:", data.model.unique())

    model_groups = data.groupby("model")
    means = list()
    stds = list()
    for model, group_df in model_groups:
        subgroup_groups = group_df.groupby("subgroup").mean()  # across seeds
        for metric in metrics:
            means.append(
                {
                    "metric": metric,
                    "model_name": model,
                    "mean_across_subgroups": subgroup_groups[metric].mean(),
                }
            )
            stds.append(
                {
                    "metric": metric,
                    "model_name": model,
                    "std_across_subgroups": subgroup_groups[metric].std(),
                }
            )

    return pd.DataFrame(means), pd.DataFrame(stds)


def bias_metrics_comparison_table(metrics, models):
    all_df = pd.concat(models)
    means, stds = mean_std_across_subgroups(all_df, metrics)
    return means.pivot_table(index="metric", columns="model_name").round(5)


def read_scores(main_dir, model_name, dataset, reg_strength=None):
    if reg_strength:
        score_files = glob.glob(
            os.path.join(main_dir, f"scores_{dataset}_{model_name}-*-{reg_strength}.pt")
        )
    else:
        score_files = glob.glob(
            os.path.join(main_dir, f"scores_{dataset}_{model_name}-*.pt")
        )
    return [torch.load(f).numpy() for f in score_files]


def compute_classification_metrics(main_dir, model_name, dataset, reg_strength=None):
    """Read scores and get classifcation metrics"""

    _, _, test = get_dataset_by_name(dataset)
    y_true = test.get_labels()

    scores = read_scores(main_dir, model_name, dataset, reg_strength)
    print(f"Found {len(scores)} scores files.")

    class_metrics = list()
    for y_pred in scores:
        class_metrics.append(evaluate_metrics(y_true, y_pred, th=0.5))

    return pd.DataFrame(class_metrics)


Results = namedtuple("Results", ["bmpi", "bm", "cm", "tm"])


def get_results(
    main_dir, model_name, bias_metrics_on=None, class_metrics_on=None, reg_strength=None
):
    """Gather all results available for a given model"""

    def attach_info(df):
        df["model_name"] = model_name
        df["bias_metrics_on"] = bias_metrics_on
        df["class_metrics_on"] = class_metrics_on
        df["reg_strength"] = reg_strength
        return df

    bias_terms_p, bias_metrics_p, class_metrics_p, test_metrics_p = (
        None,
        None,
        None,
        None,
    )
    if bias_metrics_on and reg_strength:
        bias_terms_p = f"bias_terms_{model_name}-*-{reg_strength}_{bias_metrics_on}.csv"
        bias_metrics_p = (
            f"bias_metrics_{model_name}-*-{reg_strength}_{bias_metrics_on}.csv"
        )
        class_metrics_p = (
            f"class_metrics_{model_name}-*-{reg_strength}_{bias_metrics_on}.csv"
        )

    if bias_metrics_on and not reg_strength:
        bias_terms_p = f"bias_terms_{model_name}-*_{bias_metrics_on}.csv"
        bias_metrics_p = f"bias_metrics_{model_name}-*_{bias_metrics_on}.csv"
        class_metrics_p = f"class_metrics_{model_name}-*_{bias_metrics_on}.csv"

    if class_metrics_on and reg_strength:
        test_metrics_p = (
            f"class_metrics_{model_name}-*-{reg_strength}_{class_metrics_on}.csv"
        )

    if class_metrics_on and not reg_strength:
        test_metrics_p = f"class_metrics_{model_name}-*_{class_metrics_on}.csv"

    bias_metrics_per_it, bias_metrics, class_metrics, test_metrics = (
        None,
        None,
        None,
        None,
    )

    # get bias metrics per identity term (x #seeds)
    if bias_terms_p:
        print("Get bias metrics per identity term")
        print(bias_terms_p)
        bias_metrics_per_it = match_pattern_concat(main_dir, bias_terms_p)
        bias_metrics_per_it = attach_info(bias_metrics_per_it)

    # get bias metrics
    if bias_metrics_p:
        try:
            print("Get bias metrics averaged")
            bias_metrics = match_pattern_concat(main_dir, bias_metrics_p)
            bias_metrics.columns = ["metric", "value", "seed"]
            bias_metrics = attach_info(bias_metrics)
        except:
            print(f"Files 'bias_metrics_{model_name}...' not found. Skipping...")

    # get classification metrics
    if class_metrics_p:
        try:
            print("Get classification metrics on 'bias_metrics_on' dataset")
            class_metrics = match_pattern_concat(main_dir, class_metrics_p)
            class_metrics.columns = ["metric", "value", "seed"]
            class_metrics = attach_info(class_metrics)
        except:
            print(f"Files 'class_metrics_{model_name}...' not found. Skipping...")

    if test_metrics_p:
        try:
            print("Get classification metrics on 'class_metrics_on' dataset")
            test_metrics = match_pattern_concat(main_dir, test_metrics_p)
            test_metrics.columns = ["metric", "value", "seed"]
            test_metrics = attach_info(test_metrics)
            test_metrics["metric"] = test_metrics.metric.apply(lambda x: f"test_{x}")

            #  Add summary_AUC_test
            bnsp = bias_metrics.loc[bias_metrics.metric == "bnsp_auc_mean"]
            bpsn = bias_metrics.loc[bias_metrics.metric == "bpsn_auc_mean"]
            subgroup = bias_metrics.loc[bias_metrics.metric == "subgroup_auc_mean"]
            test_AUC = test_metrics.loc[test_metrics.metric == "test_AUC"]

            #  import IPython
            #  IPython.embed()
            #  exit(-1)

            summary_AUC_test = (
                bnsp.value.values
                + bpsn.value.values
                + subgroup.value.values
                + test_AUC.value.values
            ) / 4
            bias_metrics = bias_metrics.append(
                pd.DataFrame(
                    {
                        "metric": ["summary_AUC_test"] * test_AUC.shape[0],
                        "value": summary_AUC_test,
                    }
                )
            )

        except Exception as e:
            print(
                f"Files 'class_metrics_{model_name}-*_{class_metrics_on}...' not found. Skipping...",
                e,
            )
            #  raise(e)

    return Results(bias_metrics_per_it, bias_metrics, class_metrics, test_metrics)


def show_scatter_on_metric(data: list, metrics, style="box", h_pad=2, dpi=80):
    """Create one scatter plot per dataframe in data.

    Each dataframe should contain the per-IT bias metrics of several seeds for a single model.
    """
    if not isinstance(metrics, list):
        metrics = list(metrics)

    print(f"Comparing {len(data)} model(s) on {len(metrics)} metric(s)")

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(data),
        figsize=(18, 6 * len(metrics)),
        sharey=True,
        dpi=dpi,
    )

    for i, metric in enumerate(metrics):
        for j, bias_df in enumerate(data):
            #  bias_df = bias_df.sort_values(metric)
            if style == "box":
                sns.boxplot(x="subgroup", y=metric, data=bias_df, ax=axes[i, j])
            elif style == "scatter":
                sns.stripplot(
                    x="subgroup", y=metric, data=bias_df, ax=axes[i, j], jitter=0, s=10
                )
            axes[i, j].set_title(f"{bias_df.model_name.iloc[0]}")
            axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=90)

    fig.tight_layout(h_pad=h_pad)
    return fig


def compare_metrics(data: list):
    """Create a single dataframe to compare the classification/bias metrics in data, averaged over seeds."""
    metrics_by_model = dict()
    for class_df, name in data:
        metrics_by_model[name] = class_df.groupby("metric").mean().value
    return pd.DataFrame(metrics_by_model)


def get_metrics_table(
    models,
    include_bias=True,
    include_class_eval=True,
    include_class_test=True,
    hide_power_mean=False,
):
    results = list()
    if include_bias:
        results.append(compare_metrics([(m[1].bm, m[0]) for m in models]))
    if include_class_eval:
        results.append(compare_metrics([(m[1].cm, m[0]) for m in models]))
    if include_class_test:
        results.append(
            compare_metrics([(m[1].tm, m[0]) for m in models if m[1].tm is not None])
        )

    print(len(results))
    cat = pd.concat(results)
    if hide_power_mean:
        print("hiding results with 'power_mean'")
        cat = cat.loc[[v for v in cat.index if not v.endswith("power_mean")]]
    return cat


def get_latex_tables(metric_table: pd.DataFrame):
    bias_metrics = {
        "subgroup_auc_mean": "subgroup_auc",
        "bnsp_auc_mean": "bnsp_auc",
        "bpsn_auc_mean": "bpsn_auc",
        # "positive_aeg_mean": "positive_aeg",
        # "negative_aeg_mean": "negative_aeg",
        "fped": "fped",
        "fned": "fned",
    }

    class_metrics_eval = {
        "F1_macro": "F1_macro (synt)",
        "F1_weighted": "F1_weighted (synt)",
        "F1_binary": "F1_binary (synt)",
        #  "acc": "Accuracy",
        #  "AUC": "AUC",
    }
    class_metrics_test = {
        "test_F1_macro": "F1_macro (test)",
        "test_F1_weighted": "F1_weighted (test)",
        "test_F1_binary": "F1_binary (test)",
    }

    models = {
        "vanilla": "BERT",
        "kebert_kITs": "KeBERT",
        "kebert_madITs": "KeBERT (madITs)",
        "kebert_kITsNW": "KeBERT (noW)",
        "kebert_kITsITA": "KeBERT",
        "JigCNN": "CNN",
        "JigCNN_deb": "CNN (debiased)",
        "Entropy_0.01": "EmBERT (early stop)",
        "Entropy_epoch19_0.01": "BERT+EAR",
        "BERT": "BERT",
        "BERT_EAR": "BERT+EAR",
        "BERT_bal": "BERT (class balance)",
        "BERT_EAR_bal": "BERT+EAR (class balance)",
        "BERT_SOC": "BERT_SOC"
    }

    # filter by models
    metric_table = metric_table[[m for m in models.keys() if m in metric_table.columns]]
    metric_table = metric_table.rename(columns=models)

    bias_df = metric_table.loc[bias_metrics.keys()].rename(index=bias_metrics)
    class_eval_df = metric_table.loc[class_metrics_eval.keys()].rename(
        index=class_metrics_eval
    )

    #  bias and classification performances on the evaluation set (Madlibs, Miso synt, etc.)
    eval_set_df = pd.concat([bias_df, class_eval_df], axis=0).T
    #  class. performance on the test portion (Wiki, Miso, Miso raw, etc.)

    test_df = (
        metric_table.loc[class_metrics_test.keys()].rename(index=class_metrics_test).T
    )

    return eval_set_df, test_df