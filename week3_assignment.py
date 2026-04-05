"""
Week 3 — Toxicity classification with Jigsaw dataset.

Usage:
  uv run python3 week3_assignment.py
"""

import sys
import re
import pandas as pd
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import hf_dataset, FieldSpec, Sample
from inspect_ai.solver import system_message, prompt_template, generate
from inspect_ai.scorer import model_graded_qa
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.log import EvalLog

sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "qwen2.5:7b":  "ollama/qwen2.5:7b",
    "gemma3:4b":   "ollama/gemma3:4b",
    "llama3.2:3b": "ollama/llama3.2:3b",
    "qwen3.5:9b":  "ollama/qwen3.5:9b",
    "qwen3.5:35b": "ollama/qwen3.5:35b",
}

QWEN35_CLF_CONFIG = GenerateConfig(max_tokens=50, reasoning_effort="none")
JUDGE_CONFIG = GenerateConfig(max_tokens=300)

SYSTEM_PROMPT = "Classify comments as TOXIC or NON_TOXIC."

USER_TEMPLATE = """\
Is the following comment toxic or not?

Comment:
{prompt}

LABEL: <TOXIC|NON_TOXIC>
"""

TOXICITY_SCORER_INSTRUCTIONS = """\
Given the comment and the predicted label, decide if the classification is correct.

End your response with one of:
GRADE: C
GRADE: I
"""

BLIND_TEMPLATE = """\
[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[END DATA]

{instructions}
"""

IMPROVED_SYSTEM_PROMPT = """\
You are a content moderation researcher. Classify each comment as TOXIC or NON_TOXIC.
TOXIC: insults, threats, obscenity, identity-based hate, severe personal attacks.
NON_TOXIC: civil discourse, disagreement, negative sentiment without abuse.
Respond with exactly one word: TOXIC or NON_TOXIC.
"""

IMPROVED_USER_TEMPLATE = """\
Classify this comment.

Examples:
Comment: "I think your argument is wrong and here's why..."
NON_TOXIC

Comment: "You're a worthless piece of garbage"
TOXIC

Comment: {prompt}
"""

IMPROVED_SCORER_INSTRUCTIONS = """\
You are evaluating whether a toxicity classification is correct.
- TOXIC is correct if the comment contains insults, threats, obscenity, identity attacks, or severe personal attacks.
- NON_TOXIC is correct if the comment is civil, even if critical or negative.

Respond with exactly:
GRADE: C (if correct)
GRADE: I (if incorrect)
"""


print("Loading Jigsaw dataset...")
dataset = hf_dataset(
    path="thesofakillers/jigsaw-toxic-comment-classification-challenge",
    split="train",
    sample_fields=FieldSpec(input="comment_text", target="toxic"),
)


@task
def jigsaw_toxic_binary(grade_model_name, dataset,
                        sys_prompt=SYSTEM_PROMPT, usr_template=USER_TEMPLATE,
                        scorer_instructions=TOXICITY_SCORER_INSTRUCTIONS,
                        clf_config=None):
    scorer_model = grade_model_name
    if isinstance(grade_model_name, str):
        scorer_model = get_model(grade_model_name, config=JUDGE_CONFIG)
    return Task(
        dataset,
        solver=[
            system_message(sys_prompt),
            prompt_template(usr_template),
            generate(max_tokens=50),
        ],
        scorer=model_graded_qa(
            template=BLIND_TEMPLATE,
            instructions=scorer_instructions,
            grade_pattern=r"(?is)(?:^|\n)\s*(?:GRADE\s*:\s*)?(C|I)\b",
            model=scorer_model,
        ),
        config=clf_config or GenerateConfig(),
    )


@task
def jigsaw_toxic_cheat(grade_model_name, dataset):
    scorer_model = get_model(grade_model_name, config=JUDGE_CONFIG) if isinstance(grade_model_name, str) else grade_model_name
    return Task(
        dataset,
        solver=[
            system_message(SYSTEM_PROMPT),
            prompt_template(USER_TEMPLATE),
            generate(max_tokens=50),
        ],
        scorer=model_graded_qa(
            instructions=TOXICITY_SCORER_INSTRUCTIONS,
            grade_pattern=r"(?is)(?:^|\n)\s*(?:GRADE\s*:\s*)?(C|I)\b",
            model=scorer_model,
        ),
    )


def compute_error_rates(eval_log: EvalLog) -> dict:
    clf_fp = clf_fn = clf_fail = 0
    judge_fp = judge_fn = judge_fail = 0
    total = len(eval_log.samples)

    for sample in eval_log.samples:
        gt = int(sample.target)
        completion = sample.output.completion or ""
        clf_match = re.search(r'\b(NON_TOXIC|TOXIC)\b', completion)

        if not clf_match:
            clf_fail += 1
            clf_correct = None
        else:
            predicted_toxic = clf_match.group(1) == "TOXIC"
            if predicted_toxic and gt == 0:
                clf_fp += 1
                clf_correct = False
            elif not predicted_toxic and gt == 1:
                clf_fn += 1
                clf_correct = False
            else:
                clf_correct = True

        grade = sample.scores.get("model_graded_qa")
        if grade is None:
            judge_fail += 1
            continue
        judge_val = grade.value
        if judge_val not in ("C", "I"):
            judge_fail += 1
        elif clf_correct is None:
            judge_fail += 1
        elif judge_val == "I" and clf_correct:
            judge_fp += 1
        elif judge_val == "C" and not clf_correct:
            judge_fn += 1

    return {
        "clf_fp_rate":        clf_fp / total,
        "clf_fn_rate":        clf_fn / total,
        "clf_failure_rate":   clf_fail / total,
        "judge_fp_rate":      judge_fp / total,
        "judge_fn_rate":      judge_fn / total,
        "judge_failure_rate": judge_fail / total,
    }


def toxicity_domain_score(fp_rate, fn_rate, failure_rate):
    return 1.0 - (1.0 * fp_rate + 5.0 * fn_rate + 3.0 * failure_rate)


def run_config(clf_model, judge_model, dset, n=40, label="",
               sys_prompt=SYSTEM_PROMPT, usr_template=USER_TEMPLATE,
               scorer_instructions=TOXICITY_SCORER_INSTRUCTIONS):
    clf_config = QWEN35_CLF_CONFIG if "qwen3.5" in clf_model else GenerateConfig()
    results = inspect_eval(
        jigsaw_toxic_binary(
            grade_model_name=judge_model,
            dataset=dset,
            sys_prompt=sys_prompt,
            usr_template=usr_template,
            scorer_instructions=scorer_instructions,
            clf_config=clf_config,
        ),
        model=clf_model,
        limit=n,
    )
    rates = compute_error_rates(results[0])
    tag = label or f"{clf_model} x {judge_model}"
    print(f"  {tag}: FP={rates['clf_fp_rate']:.2f} FN={rates['clf_fn_rate']:.2f} "
          f"Fail={rates['clf_failure_rate']:.2f} | "
          f"J-FP={rates['judge_fp_rate']:.2f} J-FN={rates['judge_fn_rate']:.2f} "
          f"J-Fail={rates['judge_failure_rate']:.2f}")
    return rates, results


if __name__ == "__main__":
    work_data = dataset[6:]

    # Assignment 1: Verify blind judge
    print("\n[1] Verify judge blindness")
    res_blind = inspect_eval(
        jigsaw_toxic_binary(MODELS["qwen2.5:7b"], work_data),
        model=MODELS["qwen2.5:7b"], limit=1,
    )
    res_cheat = inspect_eval(
        jigsaw_toxic_cheat(MODELS["qwen2.5:7b"], work_data),
        model=MODELS["qwen2.5:7b"], limit=1,
    )

    def get_judge_prompt(results):
        grading = results[0].samples[0].scores["model_graded_qa"].metadata["grading"]
        return grading[0]["content"]

    blind_prompt = get_judge_prompt(res_blind)
    cheat_prompt = get_judge_prompt(res_cheat)
    print(f"  Blind has target: {'target' in blind_prompt.lower() or '[Criterion]' in cheat_prompt}")
    print(f"  Cheat has [Criterion]: {'[Criterion]' in cheat_prompt or 'criterion' in cheat_prompt.lower()}")

    # Assignment 2: compute_error_rates test
    print("\n[2] compute_error_rates")
    test_res = inspect_eval(
        jigsaw_toxic_binary(MODELS["qwen2.5:7b"], work_data),
        model=MODELS["qwen2.5:7b"], limit=5,
    )
    rates_test = compute_error_rates(test_res[0])
    assert set(rates_test) == {
        "clf_fp_rate", "clf_fn_rate", "clf_failure_rate",
        "judge_fp_rate", "judge_fn_rate", "judge_failure_rate",
    }
    assert all(0.0 <= v <= 1.0 for v in rates_test.values())
    print(f"  OK: {rates_test}")

    # Assignment 3: Model comparison grid
    print("\n[3] Model comparison grid (8 configs x 40 samples)")
    configs = [
        (MODELS["qwen2.5:7b"],  MODELS["qwen2.5:7b"],  "qwen2.5:7b x qwen2.5:7b"),
        (MODELS["qwen2.5:7b"],  MODELS["gemma3:4b"],    "qwen2.5:7b x gemma3:4b"),
        (MODELS["qwen2.5:7b"],  MODELS["llama3.2:3b"],  "qwen2.5:7b x llama3.2:3b"),
        (MODELS["gemma3:4b"],   MODELS["qwen2.5:7b"],   "gemma3:4b x qwen2.5:7b"),
        (MODELS["gemma3:4b"],   MODELS["gemma3:4b"],     "gemma3:4b x gemma3:4b"),
        (MODELS["llama3.2:3b"], MODELS["qwen2.5:7b"],   "llama3.2:3b x qwen2.5:7b"),
        (MODELS["qwen3.5:9b"],  MODELS["qwen2.5:7b"],   "qwen3.5:9b x qwen2.5:7b"),
        (MODELS["qwen3.5:35b"], MODELS["qwen2.5:7b"],   "qwen3.5:35b x qwen2.5:7b"),
    ]
    grid_results = {}
    for clf, judge, label in configs:
        rates, _ = run_config(clf, judge, work_data, n=40, label=label)
        grid_results[label] = rates

    # Assignment 4: Prompt engineering
    print("\n[4] Prompt engineering")
    worst_configs = sorted(grid_results.items(),
                          key=lambda x: x[1]["clf_failure_rate"] + x[1]["judge_failure_rate"],
                          reverse=True)[:3]
    print(f"  Worst configs: {[c[0] for c in worst_configs]}")

    print("\n  [4A] Improved classifier prompt")
    for label, old_rates in worst_configs:
        clf_model = label.split(" x ")[0]
        judge_model = label.split(" x ")[1]
        clf_full = next(v for k, v in MODELS.items() if k == clf_model)
        judge_full = next(v for k, v in MODELS.items() if k == judge_model)
        new_rates, _ = run_config(
            clf_full, judge_full, work_data, n=40,
            label=f"{label} (improved clf)",
            sys_prompt=IMPROVED_SYSTEM_PROMPT,
            usr_template=IMPROVED_USER_TEMPLATE,
        )

    print("\n  [4B] Improved judge prompt")
    for label, old_rates in worst_configs:
        clf_model = label.split(" x ")[0]
        judge_model = label.split(" x ")[1]
        clf_full = next(v for k, v in MODELS.items() if k == clf_model)
        judge_full = next(v for k, v in MODELS.items() if k == judge_model)
        new_rates, _ = run_config(
            clf_full, judge_full, work_data, n=40,
            label=f"{label} (improved both)",
            sys_prompt=IMPROVED_SYSTEM_PROMPT,
            usr_template=IMPROVED_USER_TEMPLATE,
            scorer_instructions=IMPROVED_SCORER_INSTRUCTIONS,
        )

    # Assignment 5: Best pair on 200 samples
    print("\n[5] Best pair on 200 samples")
    rates_200, _ = run_config(
        MODELS["qwen2.5:7b"], MODELS["qwen2.5:7b"], work_data, n=200,
        label="qwen2.5:7b x qwen2.5:7b (n=200, improved)",
        sys_prompt=IMPROVED_SYSTEM_PROMPT,
        usr_template=IMPROVED_USER_TEMPLATE,
        scorer_instructions=IMPROVED_SCORER_INSTRUCTIONS,
    )

    # Assignment 6: Domain score
    print("\n[6] Domain score (children's platform: FN=5x, Fail=3x, FP=1x)")
    scored = []
    for label, rates in grid_results.items():
        score = toxicity_domain_score(
            rates["clf_fp_rate"], rates["clf_fn_rate"], rates["clf_failure_rate"]
        )
        scored.append((label, score))
        print(f"  {label}: {score:.3f}")
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"  Best: {scored[0][0]} ({scored[0][1]:.3f})")

    print("\nDone.")
    sys.exit(0)
