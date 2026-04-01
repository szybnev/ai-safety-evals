"""
Week 1 — Inspect AI assignments.

Usage:
  export INSPECT_MODEL="ollama/qwen3.5:35b"
  uv run python3 week1_assignment.py
"""

import os
import random
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact, match, choice
from inspect_ai.solver import (
    generate, system_message, chain_of_thought,
    prompt_template, multiple_choice
)

MODEL = os.environ.get("INSPECT_MODEL", "ollama/llama2")
random.seed(42)


@task
def hello_model():
    return Task(
        dataset=[
            Sample(input="Say 'Hello world!' and nothing else.", target="Hello world!"),
            Sample(input="2+2=", target="4"),
            Sample(input="What is the surname of Sheldon from The Big Bang Theory?", target="Cooper"),
            Sample(input="What is the capital of France? Answer with one word.", target="Paris"),
        ],
        solver=[system_message("Answer concisely and precisely."), generate()],
        scorer=match(),
    )


@task
def sentiment_classification():
    return Task(
        dataset=[
            Sample(input="I absolutely love this product! Best purchase ever.", target="Positive"),
            Sample(input="This is the worst experience I've had. Total waste of money.", target="Negative"),
            Sample(input="The package arrived on Tuesday. It was a standard box.", target="Neutral"),
            Sample(input="The movie exceeded all my expectations, incredible performances!", target="Positive"),
            Sample(input="I'm disappointed. The quality is much worse than advertised.", target="Negative"),
            Sample(input="The meeting is scheduled for 3 PM in room 204.", target="Neutral"),
        ],
        solver=[
            system_message(
                "You are a sentiment classifier. "
                "Classify the following text as exactly one of: Positive, Negative, Neutral. "
                "Reply with only the class label, nothing else."
            ),
            generate()
        ],
        scorer=exact(),
    )


def generate_questions(n: int) -> list[tuple[str, str]]:
    problems = []
    for _ in range(n):
        op = random.choice(['+', '-', '*'])
        if op == '+':
            a, b = random.randint(1, 50), random.randint(1, 50)
            answer = a + b
        elif op == '-':
            a = random.randint(10, 100)
            b = random.randint(1, a)
            answer = a - b
        else:
            a, b = random.randint(2, 15), random.randint(2, 15)
            answer = a * b
        problems.append((f"What is {a} {op} {b}?", str(answer)))
    return problems


def generate_distractors(correct: str, n: int = 3) -> list[str]:
    distractors = set()
    correct_num = int(correct)
    offsets = [-10, -5, -2, -1, 1, 2, 5, 10, -3, 3, -7, 7]

    for offset in offsets:
        if len(distractors) >= n:
            break
        candidate = str(correct_num + offset)
        if candidate != correct and candidate not in distractors:
            distractors.add(candidate)

    i = 11
    while len(distractors) < n:
        candidate = str(correct_num + i)
        if candidate != correct:
            distractors.add(candidate)
        i += 1

    return list(distractors)[:n]


def create_samples(
    questions: list[tuple[str, str]],
    correct_position: int | None = None
) -> list[Sample]:
    samples = []
    letters = ["A", "B", "C", "D"]

    for question_text, correct_answer in questions:
        distractors = generate_distractors(correct_answer, n=3)
        pos = correct_position if correct_position is not None else random.randint(0, 3)

        choices = list(distractors)
        choices.insert(pos, correct_answer)

        samples.append(Sample(
            input=question_text,
            choices=choices,
            target=letters[pos],
            metadata={"correct_answer": correct_answer, "correct_position": pos},
        ))

    return samples


@task
def position_bias_task(
    questions: list[tuple[str, str]],
    correct_position: int | None = None
):
    return Task(
        dataset=create_samples(questions, correct_position),
        solver=[multiple_choice()],
        scorer=choice(),
    )


if __name__ == "__main__":
    print(f"Модель: {MODEL}")
    print("=" * 60)

    print("\nЗадание 1: Hello World eval")
    inspect_eval(hello_model, model=MODEL)

    print("\nЗадание 2: Классификатор тональности")
    inspect_eval(sentiment_classification, model=MODEL)

    print("\nЗадание 3: Анализ позиционного смещения")
    N_QUESTIONS = 20
    random.seed(42)
    questions = generate_questions(N_QUESTIONS)

    print("Смещённый набор (ответ всегда на позиции A)...")
    inspect_eval(position_bias_task, model=MODEL, task_args={"questions": questions, "correct_position": 0})

    print("Несмещённый набор (случайная позиция)...")
    inspect_eval(position_bias_task, model=MODEL, task_args={"questions": questions, "correct_position": None})

    print("\nГотово. Запусти `uv run inspect view` для просмотра логов.")
