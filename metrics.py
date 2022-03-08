"""
Part of the code is adapted from:
- https://github.com/conversationai/unintended-ml-bias-analysis
"""
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import numpy as np
import pandas as pd
import re
import scipy.stats as stats
import logging
import numpy as np


logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def threshold_scores(scores, th: float = 0.5):
    scores = np.array(scores)
    s = np.zeros(scores.shape[0])
    s[scores >= th] = 1
    return s


def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def F1(y_true, y_pred, **kwargs):
    """Note: by default F1 is computed on the positive class."""
    return f1_score(y_true, y_pred, **kwargs)


def evaluate_metrics(y_true, y_score, th=None):
    """Evaluate multiple metrics of interest with default parameters at once."""
    perf = dict()

    # compute metrics based on scores
    perf["AUC"] = AUC(y_true, y_score)

    # compute metrics based on predictions
    y_pred = None
    if th:
        y_pred = threshold_scores(y_score, th)
        perf["acc"] = accuracy(y_true, y_pred)
        perf["F1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        perf["F1_macro"] = f1_score(y_true, y_pred, average="macro")
        perf["F1_binary"] = f1_score(y_true, y_pred, average="binary")
        perf["precision_1"] = precision_score(y_true, y_pred, pos_label=1)
        perf["precision_0"] = precision_score(y_true, y_pred, pos_label=0)
        perf["recall_1"] = recall_score(y_true, y_pred, pos_label=1)
        perf["recall_0"] = recall_score(y_true, y_pred, pos_label=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        perf["FPR"] = fp / (fp + tn)
        perf["FNR"] = fn / (fn + tp)

    return perf, y_pred


# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation
def power_mean(x, p: int, ignore_nans: bool = False):
    """Evaluate the power mean.

    If x.ndim == 1:
        x : array_like (n_rows,)
        return: float
    If x.ndim == 2:
        x : array_like (n_rows, n_cols)
        return: array_like (n_cols, )
    """
    x = np.array(x)
    mean_f = np.nanmean if ignore_nans else np.mean

    if x.ndim == 1:
        return mean_f(x ** p) ** (1 / p)
    elif x.ndim == 2:
        return mean_f(x ** p, axis=0) ** (1 / p)
    else:
        raise ValueError("The input array must be either 1D or 2D.")


# Code from:
# https://github.com/conversationai/unintended-ml-bias-analysis/model_bias_analysis.py

# Bias metrics computed for each subgroup.
SUBGROUP_SIZE = "test_size"
SUBGROUP = "subgroup"
SUBGROUP_AUC = "subgroup_auc"
NEGATIVE_CROSS_AUC = "bpsn_auc"
POSITIVE_CROSS_AUC = "bnsp_auc"
NEGATIVE_AEG = "negative_aeg"
POSITIVE_AEG = "positive_aeg"
NEGATIVE_ASEG = "negative_aseg"
POSITIVE_ASEG = "positive_aseg"
FPR = "fpr"
FPR_GAP = "fpr_gap"
FNR = "fnr"
FNR_GAP = "fnr_gap"


def add_subgroup_columns_from_text(
    df, text_column, subgroups, expect_spaces_around_words=True
):
    """Adds a boolean column for each subgroup to the data frame.

    New column contains True if the text contains that subgroup term.

    Args:
      df: Pandas dataframe to process.
      text_column: Column in df containing the text.
      subgroups: List of subgroups to search text_column for.
      expect_spaces_around_words: Whether to expect subgroup to be surrounded by
        spaces in the text_column.  Set to False to for languages which do not
        use spaces.
    """
    ndf = df.copy()
    for term in subgroups:
        if expect_spaces_around_words:
            # pylint: disable=cell-var-from-loop
            ndf[term] = ndf[text_column].apply(
                lambda x: bool(
                    re.search("\\b" + term + "\\b", x, flags=re.UNICODE | re.IGNORECASE)
                )
            )
        else:
            ndf[term] = ndf[text_column].str.contains(term, case=False)
    return ndf


def compute_bias_metrics_for_subgroup_and_model(
    dataset: pd.DataFrame,
    subgroup: str,
    model: str,
    label_col: str,
    threshold: float = 0.5,
    include_asegs=False,
):
    """Computes per-subgroup metrics for one model and subgroup.

    This the general method to extend if new metrics are included/excluded.
    """
    record = {SUBGROUP: subgroup, SUBGROUP_SIZE: len(dataset[dataset[subgroup]])}
    record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
    record[NEGATIVE_CROSS_AUC] = compute_negative_cross_auc(
        dataset, subgroup, label_col, model
    )
    record[POSITIVE_CROSS_AUC] = compute_positive_cross_auc(
        dataset, subgroup, label_col, model
    )
    record[NEGATIVE_AEG] = compute_negative_aeg(dataset, subgroup, label_col, model)
    record[POSITIVE_AEG] = compute_positive_aeg(dataset, subgroup, label_col, model)

    record[FPR] = compute_fpr(dataset, label_col, model, threshold, subgroup)
    record[FPR_GAP] = compute_fpr(dataset, label_col, model, threshold) - record[FPR]
    record[FNR] = compute_fnr(dataset, label_col, model, threshold, subgroup)
    record[FNR_GAP] = compute_fnr(dataset, label_col, model, threshold) - record[FNR]

    if include_asegs:
        (
            record[POSITIVE_ASEG],
            record[NEGATIVE_ASEG],
        ) = compute_average_squared_equality_gap(dataset, subgroup, label_col, model)
    return record


def column_name(model, metric):
    return f"{model}_{metric}"


###################################
#  AUC-based metrics (Borkan et al., 2019)
###################################


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    try:
        return AUC(subgroup_examples[label], subgroup_examples[model_name])
    except ValueError as e:
        logger.error(
            f"Trying to compute AUC on subgroup {subgroup}: {e}. Returning np.nan"
        )
        return np.nan


def compute_negative_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    try:
        return AUC(examples[label], examples[model_name])
    except ValueError as e:
        logger.error(
            f"Trying to compute AUC on subgroup {subgroup}: {e}. Returning np.nan"
        )
        return np.nan


def compute_positive_cross_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    try:
        return AUC(examples[label], examples[model_name])
    except ValueError as e:
        logger.error(
            f"Trying to compute AUC on subgroup {subgroup}: {e}. Returning np.nan"
        )
        return np.nan


###################################
#  Threshold independent metrics (Borkan et al., 2019)
###################################


def normalized_mwu(data1, data2, model_name):
    """Calculate number of datapoints with a higher score in data1 than data2."""
    scores_1 = data1[model_name]
    scores_2 = data2[model_name]
    n1 = len(scores_1)
    n2 = len(scores_2)
    if n1 == 0 or n2 == 0:
        return None
    u, _ = stats.mannwhitneyu(scores_1, scores_2, alternative="less")
    return u / (n1 * n2)


def compute_negative_aeg(df, subgroup, label, model_name):
    """Compute the Negative Average Equality Gap"""
    mwu = normalized_mwu(
        df[~df[subgroup] & ~df[label]], df[df[subgroup] & ~df[label]], model_name
    )
    if mwu is None:
        return np.nan
    return 0.5 - mwu


def compute_positive_aeg(df, subgroup, label, model_name):
    """Compute the Positive Average Equality Gap"""
    mwu = normalized_mwu(
        df[~df[subgroup] & df[label]], df[df[subgroup] & df[label]], model_name
    )
    if mwu is None:
        return np.nan
    return 0.5 - mwu


def compute_average_squared_equality_gap(df, subgroup, label, model_name):
    """Returns the positive and negative ASEG metrics.

    TODO: extend this definition
    """
    subgroup_df = df[df[subgroup]]
    background_df = df[~df[subgroup]]
    if subgroup_df.empty or background_df.empty:
        return np.nan, np.nan

    thresholds = np.linspace(1.0, 0.0, num=1000)
    s_fpr, s_tpr = positive_rates(subgroup_df, model_name, label, thresholds)
    b_fpr, b_tpr = positive_rates(background_df, model_name, label, thresholds)

    def squared_diff_integral(y, x):
        return np.trapz(np.square(np.subtract(y, x)), x)

    if s_fpr and s_tpr and b_fpr and b_tpr:
        return squared_diff_integral(s_tpr, b_tpr), squared_diff_integral(s_fpr, b_fpr)
    return np.nan, np.nan


def positive_rates(df, score_col, label_col, thresholds):
    """Compute false positive and true positive rates."""
    tpr = []
    fpr = []
    for threshold in thresholds:
        confusion = confusion_matrix_counts(df, score_col, label_col, threshold)
        if (
            confusion["tp"] + confusion["fn"] == 0
            or confusion["fp"] + confusion["tn"] == 0
        ):
            return None, None
        tpr.append(confusion["tp"] / (confusion["tp"] + confusion["fn"]))
        fpr.append(confusion["fp"] / (confusion["fp"] + confusion["tn"]))
    return fpr, tpr


def confusion_matrix_counts(df, score_col, label_col, threshold):
    return {
        "tp": len(df[(df[score_col] >= threshold) & df[label_col]]),
        "tn": len(df[(df[score_col] < threshold) & ~df[label_col]]),
        "fp": len(df[(df[score_col] >= threshold) & ~df[label_col]]),
        "fn": len(df[(df[score_col] < threshold) & df[label_col]]),
    }


###################################
#  Error Rate Equality Difference (or Equality of Odds, Hardt, 2016)
###################################


def false_positive_equality_difference(
    df: pd.DataFrame, label_col: str, scores_col: str, threshold: float, subgroups: list
):
    """Compute False Positive Equality Difference."""
    fpr = compute_fpr(df, label_col, scores_col, threshold)
    subg_fprs = np.array(
        [compute_fpr(df, label_col, scores_col, threshold, subg) for subg in subgroups]
    )
    return (fpr - subg_fprs).abs().sum()


def false_negative_equality_difference(
    df: pd.DataFrame, label_col: str, scores_col: str, threshold: float, subgroups: list
):
    """Compute False Negative Equality Difference."""
    fnr = compute_fnr(df, label_col, scores_col, threshold)
    subg_fnrs = np.array(
        [compute_fnr(df, label_col, scores_col, threshold, subg) for subg in subgroups]
    )
    return (fnr - subg_fnrs).abs().sum()


def compute_fpr(df, label, model_name, threshold, subgroup: str = None):
    """Compute FPR (optionally on a subgroup)."""
    if subgroup:
        df = df[df[subgroup]]
    cm = confusion_matrix_counts(df, model_name, label, threshold)
    return cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"] != 0) else np.nan


def compute_fnr(df, label, model_name, threshold, subgroup: str = None):
    """Compute FNR (optionally on a subgroup)."""
    if subgroup:
        df = df[df[subgroup]]
    cm = confusion_matrix_counts(df, model_name, label, threshold)
    return cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"] != 0) else np.nan
