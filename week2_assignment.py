"""
Week 2 — MMLU evaluation with statistical analysis.

Usage:
  uv run python3 week2_assignment.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from string import ascii_uppercase
from typing import Tuple, List

from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import multiple_choice, chain_of_thought
from inspect_ai.scorer import choice
from inspect_ai.log import EvalLog

sys.stdout.reconfigure(line_buffering=True)

MODEL_A = os.environ.get("MODEL_A", "ollama/qwen3.5:9b")
MODEL_B = os.environ.get("MODEL_B", "ollama/qwen3.5:35b")


def record_to_sample(record: dict) -> Sample:
    answer_idx = int(record["answer"])
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=ascii_uppercase[answer_idx],
        metadata=dict(subject=record.get("subject"))
    )


dataset = hf_dataset(
    path="cais/mmlu", name="all", split="test",
    sample_fields=record_to_sample, cached=True
)

MY_SUBJECT = "computer_security"
MY_SUBSET = dataset.filter(lambda s: s.metadata.get("subject") == MY_SUBJECT)


@task
def mmlu_subset(subset):
    return Task(dataset=subset, solver=[multiple_choice()], scorer=choice())


@task
def mmlu_subset_cot(subset):
    return Task(dataset=subset, solver=[chain_of_thought(), multiple_choice()], scorer=choice())


def log_to_df(log: EvalLog) -> pd.DataFrame:
    rows = []
    for sample in log.samples:
        score_val = sample.scores["choice"].value
        rows.append({
            "id": sample.id,
            "epoch": sample.epoch,
            "score": 1 if score_val == "C" else 0,
            "subject": sample.metadata.get("subject", ""),
        })
    return pd.DataFrame(rows)


def ci_accuracy_basic(scores: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    n = len(scores)
    mean = scores.mean()
    if n <= 1 or mean == 0.0 or mean == 1.0:
        return (mean, mean, mean)
    se = np.sqrt(mean * (1 - mean) / n)
    z = stats.norm.ppf((1 + ci) / 2)
    return (max(0, mean - z * se), mean, min(1, mean + z * se))


def ci_accuracy(df: pd.DataFrame, ci: float = 0.95) -> Tuple[float, float, float]:
    q_means = df.groupby("id")["score"].mean()
    return ci_accuracy_basic(q_means.values, ci)


def significance_by_paired_ttest(
    scores1: np.ndarray, scores2: np.ndarray,
    alpha: float = 0.05, two_tailed: bool = True,
) -> Tuple[float, float, bool]:
    diffs = scores1 - scores2
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    if not two_tailed:
        p_value = p_value / 2
        if t_stat < 0:
            p_value = 1 - p_value
    return (p_value, diffs.mean(), p_value < alpha)


def ci_accuracy_for_difference(
    scores1: np.ndarray, scores2: np.ndarray, ci: float = 0.95
) -> Tuple[float, float, float]:
    diffs = scores1 - scores2
    n = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf((1 + ci) / 2)
    return (mean_diff - z * se, mean_diff, mean_diff + z * se)


def estimate_variance_components(logs_a: List[EvalLog], logs_b: List[EvalLog]) -> dict:
    df_a = log_to_df(logs_a[0])
    df_b = log_to_df(logs_b[0])
    q_means_a = df_a.groupby("id")["score"].mean()
    q_means_b = df_b.groupby("id")["score"].mean()
    common_ids = sorted(set(q_means_a.index) & set(q_means_b.index))
    q_means_a = q_means_a.loc[common_ids]
    q_means_b = q_means_b.loc[common_ids]
    q_avg = (q_means_a.values + q_means_b.values) / 2
    omega2 = np.var(q_avg, ddof=1)
    sigma2_a = sigma2_b = 0.0
    for qid in common_ids:
        sa = df_a[df_a["id"] == qid]["score"].values
        sb = df_b[df_b["id"] == qid]["score"].values
        if len(sa) > 1: sigma2_a += np.var(sa, ddof=1)
        if len(sb) > 1: sigma2_b += np.var(sb, ddof=1)
    n_q = len(common_ids)
    sigma2_a /= max(n_q, 1)
    sigma2_b /= max(n_q, 1)
    return {"omega2": omega2, "sigma2_a": sigma2_a, "sigma2_b": sigma2_b}


def minimum_detectable_effect(
    n: int, omega2: float, sigma2_a: float = 0.0, sigma2_b: float = 0.0,
    ka: int = 1, kb: int = 1, alpha: float = 0.05, power: float = 0.80,
) -> float:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    var_total = omega2 + sigma2_a / ka + sigma2_b / kb
    return (z_alpha + z_beta) * np.sqrt(var_total / n)


def required_sample_size(
    delta: float, omega2: float, sigma2_a: float = 0.0, sigma2_b: float = 0.0,
    ka: int = 1, kb: int = 1, alpha: float = 0.05, power: float = 0.80,
) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    var_total = omega2 + sigma2_a / ka + sigma2_b / kb
    return int(np.ceil(((z_alpha + z_beta) ** 2 * var_total) / delta ** 2))


def run_and_get_scores(model_name: str, dset, epochs: int = 1) -> np.ndarray:
    run_logs = inspect_eval(mmlu_subset(dset), model=model_name, epochs=epochs)
    df = log_to_df(run_logs[0])
    return df.groupby("id")["score"].mean().sort_index().values


if __name__ == "__main__":
    print(f"{MODEL_A} vs {MODEL_B} | subset: {MY_SUBJECT} ({len(MY_SUBSET)}q)")

    print("\n[2] log_to_df")
    test_logs = inspect_eval(mmlu_subset(MY_SUBSET), model=MODEL_A, limit=10)
    df_test = log_to_df(test_logs[0])
    print(f"  {len(df_test)} rows, accuracy={df_test['score'].mean():.2%}")

    print("\n[3] CI")
    logs_full_a = inspect_eval(mmlu_subset(MY_SUBSET), model=MODEL_A)
    df_full_a = log_to_df(logs_full_a[0])
    lo, acc, hi = ci_accuracy(df_full_a)
    print(f"  accuracy={acc:.3f} CI=[{lo:.3f}, {hi:.3f}]")

    print("\n[4.1] CI vs epochs (k=1,2,5)")
    k_values = [1, 2, 5]
    accs_k, lo_k, hi_k = [], [], []
    for k in k_values:
        logs_k = inspect_eval(mmlu_subset(MY_SUBSET), model=MODEL_A, epochs=k)
        l, m, h = ci_accuracy(log_to_df(logs_k[0]))
        lo_k.append(l); accs_k.append(m); hi_k.append(h)
        print(f"  k={k}: acc={m:.3f} CI=[{l:.3f}, {h:.3f}]")
    plt.figure(figsize=(8, 4))
    plt.fill_between(k_values, lo_k, hi_k, alpha=0.25, label="95% CI")
    plt.plot(k_values, accs_k, "o-", lw=2, label="Accuracy")
    plt.xlabel("Epochs (K)"); plt.ylabel("Accuracy")
    plt.title(f"{MODEL_A} — CI vs epochs"); plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig("plot_ci_vs_epochs.png", dpi=100)

    print("\n[4.2] CI vs n")
    question_ids = sorted(df_full_a["id"].unique())
    dataset_sizes = range(10, len(question_ids) + 1, 10)
    accs_n, lo_n, hi_n = [], [], []
    for n in dataset_sizes:
        df_slice = df_full_a[df_full_a["id"].isin(question_ids[:n])]
        l, m, h = ci_accuracy(df_slice)
        lo_n.append(l); accs_n.append(m); hi_n.append(h)
    plt.figure(figsize=(8, 4))
    plt.fill_between(list(dataset_sizes), lo_n, hi_n, alpha=0.25, label="95% CI")
    plt.plot(list(dataset_sizes), accs_n, "o-", lw=2, label="Accuracy")
    plt.xlabel("n (questions)"); plt.ylabel("Accuracy")
    plt.title(f"{MODEL_A} — CI vs n"); plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig("plot_ci_vs_n.png", dpi=100)

    print("\n[5] Paired t-test")
    scores_a = run_and_get_scores(MODEL_A, MY_SUBSET)
    scores_b = run_and_get_scores(MODEL_B, MY_SUBSET)
    p_val, mean_diff, sig = significance_by_paired_ttest(scores_a, scores_b)
    print(f"  p={p_val:.4f} diff={mean_diff:.4f} sig={sig}")

    print("\n[6] CI on difference")
    lo_d, mean_d, hi_d = ci_accuracy_for_difference(scores_a, scores_b)
    print(f"  diff={mean_d:.4f} CI=[{lo_d:.4f}, {hi_d:.4f}] zero_in_ci={lo_d <= 0 <= hi_d}")

    print("\n[7] Variance components + MDE")
    pilot_a = inspect_eval(mmlu_subset(MY_SUBSET), model=MODEL_A, epochs=2, limit=15)
    pilot_b = inspect_eval(mmlu_subset(MY_SUBSET), model=MODEL_B, epochs=2, limit=15)
    params = estimate_variance_components(pilot_a, pilot_b)
    mde = minimum_detectable_effect(n=len(MY_SUBSET), **params)
    print(f"  omega2={params['omega2']:.4f} sigma2_a={params['sigma2_a']:.4f} sigma2_b={params['sigma2_b']:.4f}")
    print(f"  MDE(n={len(MY_SUBSET)})={mde:.1%}")

    print("\n[8] Required sample size")
    print(f"  delta=5%: n={required_sample_size(delta=0.05, **params)}")
    print(f"  delta=10%: n={required_sample_size(delta=0.10, **params)}")

    print("\n[9] Default vs CoT")
    logs_cot = inspect_eval(mmlu_subset_cot(MY_SUBSET), model=MODEL_A)
    scores_cot = log_to_df(logs_cot[0]).groupby("id")["score"].mean().sort_index().values
    min_len = min(len(scores_a), len(scores_cot))
    s_def, s_cot = scores_a[:min_len], scores_cot[:min_len]
    p_cot, diff_cot, sig_cot = significance_by_paired_ttest(s_cot, s_def)
    lo_c, m_c, hi_c = ci_accuracy_for_difference(s_cot, s_def)
    print(f"  default={s_def.mean():.3f} cot={s_cot.mean():.3f}")
    print(f"  diff={diff_cot:.4f} p={p_cot:.4f} sig={sig_cot} CI=[{lo_c:.4f}, {hi_c:.4f}]")

    print("\nГотово.")
    sys.exit(0)
