"""
Week 4 — Evaluating LLM Agents on Mathematical Reasoning (ReAct + MATH-500).

Usage:
  uv run python3 week4_assignment.py
"""

import sys
import re
import math
import random
from textwrap import dedent
from collections import defaultdict
from scipy.stats import norm

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.agent import react
from inspect_ai.solver import generate, use_tools, system_message
from inspect_ai.scorer import model_graded_qa, match
from inspect_ai.tool import tool
from inspect_ai.model import GenerateConfig, get_model

sys.stdout.reconfigure(line_buffering=True)

MODEL = "ollama/qwen2.5:7b"
SCORER_MODEL = "ollama/qwen2.5:7b"
RANDOM_SEED = 42
EVAL_LIMIT = 30
MAX_MESSAGES = 20


def get_acc(log):
    m = log.results.scores[0].metrics
    return (m.get("accuracy") or m.get("mean")).value


def _first_score(sample):
    scores = sample.scores
    first_key = list(scores.keys())[0]
    val = scores[first_key]
    return val[0] if isinstance(val, list) else val


def print_results(label, log):
    acc = get_acc(log)
    print(f"  {label}: {acc:.0%} ({len(log.samples)} samples)")


def wilson_ci(n_correct, n_total, confidence_level=0.95):
    z = norm.ppf(0.5 + confidence_level / 2)
    p_hat = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p_hat + z ** 2 / (2 * n_total)) / denom
    margin = z * math.sqrt(
        (p_hat * (1 - p_hat) + z ** 2 / (4 * n_total)) / n_total
    ) / denom
    return max(0, centre - margin), min(1, centre + margin)


# --- Tools ---

@tool
def add():
    async def execute(a: float, b: float) -> str:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        try:
            return str(float(a) + float(b))
        except Exception as e:
            return f"Error: {e}"
    return execute

@tool
def subtract():
    async def execute(a: float, b: float) -> str:
        """Subtract b from a.

        Args:
            a: Number to subtract from.
            b: Number to subtract.
        """
        try:
            return str(float(a) - float(b))
        except Exception as e:
            return f"Error: {e}"
    return execute

@tool
def multiply():
    async def execute(a: float, b: float) -> str:
        """Multiply two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        try:
            return str(float(a) * float(b))
        except Exception as e:
            return f"Error: {e}"
    return execute

@tool
def divide():
    async def execute(a: float, b: float) -> str:
        """Divide a by b.

        Args:
            a: Dividend.
            b: Divisor (must not be zero).
        """
        try:
            b_val = float(b)
            if b_val == 0:
                return "Error: division by zero."
            return str(float(a) / b_val)
        except Exception as e:
            return f"Error: {e}"
    return execute

@tool
def modular_arithmetic():
    async def execute(a: int, b: int) -> str:
        """Compute a mod b (remainder of a divided by b).

        Args:
            a: Dividend.
            b: Divisor (must not be zero).
        """
        try:
            b_val = int(b)
            if b_val == 0:
                return "Error: modulo by zero."
            return str(int(a) % b_val)
        except Exception as e:
            return f"Error: {e}"
    return execute

@tool
def sympy_solve():
    async def execute(equation: str) -> str:
        """Solve an equation symbolically using SymPy. Pass the equation set to zero, e.g. '2*x + 5 - 21'.

        Args:
            equation: Equation string set to zero.
        """
        try:
            import sympy
            expr = sympy.sympify(equation)
            solutions = sympy.solve(expr)
            return str(solutions)
        except Exception as e:
            return f"Error: {e}"
    return execute


ARITH_TOOLS = [add(), subtract(), multiply(), divide(), modular_arithmetic()]
ALL_TOOLS = ARITH_TOOLS + [sympy_solve()]


# --- Toy samples ---

TOY_SAMPLES = [
    Sample(input="A semiconductor factory produced 48,397 chips on Monday and 63,518 chips on Tuesday. How many chips were produced in total?", target="111915"),
    Sample(input="A government reserve had 874,203 barrels of oil. After an emergency release, 295,867 barrels were distributed. How many barrels remain in the reserve?", target="578336"),
    Sample(input="A logistics company ships 4,738 containers, each holding 2,659 units. How many units are shipped in total?", target="12598342"),
    Sample(input="A national census counted 8,743,291 residents across 6,473 districts. If residents are distributed equally, how many full residents are assigned per district?", target="1350"),
    Sample(input="A satellite completes a full orbit every 397 minutes. After exactly 1,000,000 minutes of operation, how many minutes have passed since the last complete orbit?", target="354"),
    Sample(input="A hospital ordered 12,475 boxes of supplies at 387 dollars per box. They received a bulk discount of 843,750 dollars off the total. How much did the hospital pay after the discount?", target="3984075"),
    Sample(input="A city has 14 times as many residents as municipal employees. If the total number of residents and employees together is 489,375, how many municipal employees does the city have?", target="32625"),
    Sample(input="An airline flew 3,847 domestic flights and 2,964 international flights last month. Each flight used an average of 8,753 liters of fuel. How many liters of fuel were used in total?", target="59616683"),
    Sample(input="A clock tower rings a bell every 1,873 seconds. After 10,000,000 seconds have elapsed since midnight, how many seconds ago did the bell last ring?", target="53"),
    Sample(input="A farm harvested 247,839 kg of wheat and 184,672 kg of barley. The grain is loaded into trucks that carry exactly 4,750 kg each. How many full truckloads can be made from all the grain?", target="91"),
    Sample(input="A global streaming platform has 1,847,293,847,291 seconds of video content. Given that a day has 86,400 seconds, how many full days of content does the platform have?", target="21380715"),
    Sample(input="A country's economy grew by 3,847 dollars per citizen in a year. The country has 847,293,847 citizens. What was the total economic growth in dollars?", target="3259539429409"),
]


# --- MATH-500 setup ---

TOOL_SUBJECTS = ["Algebra", "Number Theory", "Prealgebra", "Intermediate Algebra"]

def extract_boxed(solution):
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return solution.strip()
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{": depth += 1
        elif solution[i] == "}": depth -= 1
        i += 1
    return solution[start:i - 1].strip()

def record_to_sample(record):
    target = record.get("answer") or extract_boxed(record["solution"])
    return Sample(
        input=record["problem"],
        target=target,
        metadata={"level": int(record["level"]), "subject": record["subject"]},
    )


# --- Assignment 3: Scorer ---

GRADING_INSTRUCTIONS = dedent("""\
    Compare the submitted answer with the expected answer.
    Two answers are equivalent if they represent the same mathematical value,
    even if written differently (e.g. 10/4 and 5/2, or 0.5 and 1/2).
    Grade C if the answers are mathematically equivalent, I otherwise.
    End with: GRADE: C or GRADE: I
""")

MATH_SCORER = model_graded_qa(
    model=SCORER_MODEL,
    instructions=GRADING_INSTRUCTIONS,
)


# --- Assignment 4: ReAct prompts ---

SIMPLE_PROMPT = dedent("""\
    You are a math solver. Read the problem carefully, compute the answer,
    and respond with the final numeric result.
""")

NAIVE_LOOP_PROMPT = dedent("""\
    You are a math solver with access to calculator tools.
    Break each problem into arithmetic steps and call one tool per step.
""")

REACT_PROMPT_V1 = dedent("""\
    You are a math solver with access to calculator tools.
    Break each problem into arithmetic steps and call one tool per step.
    Don't calculate anything without tools.
    After getting the final numeric result, call submit() with ONLY the number.
""")

REACT_PROMPT_V2 = dedent("""\
    You are a precise math solver. You have calculator and algebra tools.
    Rules:
    1. Break every problem into small arithmetic steps.
    2. Use tools for ALL calculations, never compute in your head.
    3. For modular arithmetic (remainders), use modular_arithmetic(a, b).
    4. For equations, use sympy_solve(equation_set_to_zero).
    5. After getting the final answer, call submit() with ONLY the number.
    6. If the answer should be an integer, round or truncate as needed.
    Think step by step, then act.
""")

REACT_PROMPT_V3 = dedent("""\
    You are a precise math solver with calculator and algebra tools.
    ALWAYS use tools for calculations. NEVER compute mentally.
    Steps:
    1. Read the problem and identify what operations are needed.
    2. Call tools one at a time: add, subtract, multiply, divide, modular_arithmetic, sympy_solve.
    3. After each tool result, verify it makes sense before proceeding.
    4. When you have the final numeric answer, call submit(answer).
    Important: submit() expects ONLY a number (integer or decimal). No units, no text.
""")


if __name__ == "__main__":
    print(f"Model: {MODEL}")

    # Load MATH-500
    print("\nLoading MATH-500...")
    full_dataset = hf_dataset(
        path="HuggingFaceH4/MATH-500", split="test",
        sample_fields=record_to_sample, cached=True,
    )
    tool_dataset = [s for s in full_dataset if s.metadata["subject"] in TOOL_SUBJECTS]
    random.seed(RANDOM_SEED)
    random.shuffle(tool_dataset)
    split_point = int(len(tool_dataset) * 0.1)
    DEV_SET = tool_dataset[:split_point]
    TEST_SET = tool_dataset[split_point:]
    print(f"  Tool-friendly: {len(tool_dataset)}, DEV: {len(DEV_SET)}, TEST: {len(TEST_SET)}")

    # Toy: Approach 0 — generate only
    print("\n[Toy] Approach 0: generate only")
    log_gen = inspect_eval(Task(
        dataset=TOY_SAMPLES,
        solver=[system_message(SIMPLE_PROMPT), generate()],
        scorer=match(numeric=True),
    ), model=MODEL)[0]
    print_results("generate only", log_gen)

    # Toy: Approach A — naive tool loop
    print("\n[Toy] Approach A: naive tool loop")
    log_naive = inspect_eval(Task(
        dataset=TOY_SAMPLES,
        solver=[system_message(NAIVE_LOOP_PROMPT), use_tools(ARITH_TOOLS), generate()],
        scorer=match(numeric=True),
    ), model=MODEL)[0]
    print_results("naive tool loop", log_naive)

    # Toy: Approach B — ReAct v1
    print("\n[Toy] Approach B: ReAct v1")
    log_react = inspect_eval(Task(
        dataset=TOY_SAMPLES,
        solver=react(prompt=REACT_PROMPT_V1, tools=ARITH_TOOLS, attempts=1),
        scorer=match(numeric=True),
        message_limit=MAX_MESSAGES,
    ), model=MODEL)[0]
    print_results("react v1", log_react)

    # Toy comparison
    print("\n[Toy] Comparison:")
    for label, log in [("generate", log_gen), ("naive", log_naive), ("react v1", log_react)]:
        acc = get_acc(log)
        msgs = [len(s.messages) for s in log.samples]
        print(f"  {label:<20s} acc={acc:.0%} avg_msgs={sum(msgs)/len(msgs):.1f}")

    # Assignment 3: scorer test
    print("\n[3] Scorer test")
    log_scorer = inspect_eval(Task(
        dataset=[
            Sample(input="What is 1+1?", target="2"),
            Sample(input="What is 10/4?", target="5/2"),
        ],
        solver=[system_message("Answer the math question. Reply with just the answer."), generate()],
        scorer=MATH_SCORER,
    ), model=MODEL)[0]
    print_results("scorer test", log_scorer)

    # Assignment 4: dev-set iteration
    print("\n[4] Dev-set iteration")
    DEV_RUNS = []

    print("  Attempt 1 (react v1)...")
    log_a1 = inspect_eval(Task(
        dataset=DEV_SET,
        solver=react(prompt=REACT_PROMPT_V1, tools=ALL_TOOLS, attempts=1),
        scorer=MATH_SCORER, message_limit=MAX_MESSAGES,
    ), model=MODEL, limit=EVAL_LIMIT)[0]
    print_results("attempt 1 (v1)", log_a1)
    DEV_RUNS.append(("attempt 1 (v1)", log_a1))

    print("  Attempt 2 (react v2 — detailed rules)...")
    log_a2 = inspect_eval(Task(
        dataset=DEV_SET,
        solver=react(prompt=REACT_PROMPT_V2, tools=ALL_TOOLS, attempts=1),
        scorer=MATH_SCORER, message_limit=MAX_MESSAGES,
    ), model=MODEL, limit=EVAL_LIMIT)[0]
    print_results("attempt 2 (v2)", log_a2)
    DEV_RUNS.append(("attempt 2 (v2)", log_a2))

    print("  Attempt 3 (react v3 — verify steps)...")
    log_a3 = inspect_eval(Task(
        dataset=DEV_SET,
        solver=react(prompt=REACT_PROMPT_V3, tools=ALL_TOOLS, attempts=1),
        scorer=MATH_SCORER, message_limit=MAX_MESSAGES,
    ), model=MODEL, limit=EVAL_LIMIT)[0]
    print_results("attempt 3 (v3)", log_a3)
    DEV_RUNS.append(("attempt 3 (v3)", log_a3))

    print("\n  Dev summary:")
    best_log = None
    best_acc = -1
    best_name = ""
    for name, log in DEV_RUNS:
        acc = get_acc(log)
        print(f"    {name:<30s} {acc:.0%}")
        if acc > best_acc:
            best_acc = acc
            best_log = log
            best_name = name

    # Assignment 5.1: test-set eval
    print(f"\n[5.1] Test-set eval (best: {best_name})")
    best_prompt = REACT_PROMPT_V1
    if "v2" in best_name: best_prompt = REACT_PROMPT_V2
    if "v3" in best_name: best_prompt = REACT_PROMPT_V3

    log_test = inspect_eval(Task(
        dataset=TEST_SET,
        solver=react(prompt=best_prompt, tools=ALL_TOOLS, attempts=1),
        scorer=MATH_SCORER, message_limit=MAX_MESSAGES,
    ), model=MODEL, limit=len(TEST_SET))[0]

    n_test = len(log_test.samples)
    n_correct = sum(1 for s in log_test.samples if _first_score(s).value == "C")
    lo, hi = wilson_ci(n_correct, n_test)
    print(f"  Test accuracy: {n_correct/n_test:.1%}")
    print(f"  95% Wilson CI: [{lo:.1%}, {hi:.1%}]")
    print(f"  n = {n_test}")

    # Assignment 5.2: breakdown
    print("\n[5.2] Breakdown by subject and level")
    subject_stats = defaultdict(lambda: [0, 0])
    level_stats = defaultdict(lambda: [0, 0])
    for sample in log_test.samples:
        sc = _first_score(sample)
        correct = 1 if sc.value == "C" else 0
        subject_stats[sample.metadata.get("subject", "?")][0] += correct
        subject_stats[sample.metadata.get("subject", "?")][1] += 1
        level_stats[sample.metadata.get("level", 0)][0] += correct
        level_stats[sample.metadata.get("level", 0)][1] += 1

    print(f"  {'Subject':<25s} {'Correct':>7s} {'Total':>5s} {'Acc':>6s}")
    for subj in sorted(subject_stats):
        c, t = subject_stats[subj]
        print(f"  {subj:<25s} {c:>7d} {t:>5d} {c/t:>6.0%}")

    print(f"\n  {'Level':<25s} {'Correct':>7s} {'Total':>5s} {'Acc':>6s}")
    for lvl in sorted(level_stats):
        c, t = level_stats[lvl]
        print(f"  Level {lvl:<18d} {c:>7d} {t:>5d} {c/t:>6.0%}")

    # Dev vs Test
    print(f"\n  Dev vs Test:")
    for name, log in DEV_RUNS:
        print(f"    {name:<30s} dev={get_acc(log):.0%}")
    print(f"    {'best (TEST)':<30s} test={n_correct/n_test:.1%}")

    print("\nDone.")
    sys.exit(0)
