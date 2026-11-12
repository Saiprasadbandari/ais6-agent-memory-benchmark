"""Shared LLM-as-a-Judge (GPT-4o-mini) & 'Lost vs. Ignored' evidence diagnostic."""

from typing import Callable, List


JUDGE_RUBRIC_PROMPT = """You are an impartial evaluator grading an AI assistant's response on the LongMemEval benchmark.

[Question]: {question}
[Question Type]: {question_type}
[Gold Answer]: {gold_answer}
[Model Prediction]: {prediction}

Grading Rules:
1. If Question Type is 'abstention' (or the question asks about unmentioned facts):
   - Score 1 if the assistant rightly abstains / states the information was never mentioned.
   - Score 0 if the assistant hallucinates or guesses an answer.
2. For all other question types:
   - Score 1 if the prediction contains the core factual information in the Gold Answer (exact wording not required, but dates, numbers, and updated states must match).
   - Score 0 if the prediction is incorrect, outdated, or missing key facts.

Respond with ONLY '1' or '0'."""


EVIDENCE_SURVIVAL_PROMPT = """You are auditing an AI memory system ("Lost vs. Ignored" error analysis).
Check whether the critical ground-truth evidence required to answer the question survived inside the memory context below.

[Question]: {question}
[Ground-Truth Evidence]: {evidence}
[Constructed Memory Context]:
{memory_context}

Did the core fact/information in [Ground-Truth Evidence] survive in [Constructed Memory Context]?
Respond with ONLY 'PRESENT' or 'ABSENT'."""


def evaluate_answer_correctness(
    question: str,
    question_type: str,
    gold_answer: str,
    prediction: str,
    judge_call_fn: Callable[[str], str],
) -> int:
    """Grade an answer (1 = correct, 0 = wrong) using the shared LLM judge protocol."""
    prompt = JUDGE_RUBRIC_PROMPT.format(
        question=question,
        question_type=question_type,
        gold_answer=gold_answer,
        prediction=prediction,
    )
    raw = judge_call_fn(prompt).strip()
    return 1 if raw.startswith("1") else 0


def diagnose_lost_vs_ignored(
    is_correct: int,
    question: str,
    evidence_list: List[str],
    memory_context: str,
    judge_call_fn: Callable[[str], str] | None = None,
) -> str:
    """Classify wrong answers into 'lost_by_memory' vs 'ignored_by_reader'.

    Returns one of: 'correct', 'lost_by_memory', 'ignored_by_reader'.
    """
    if is_correct == 1:
        return "correct"

    evidence_joined = " | ".join(evidence_list)
    if not evidence_joined.strip():
        return "ignored_by_reader"

    # Fast lexical substring check first
    mem_lower = memory_context.lower()
    if any(ev.lower() in mem_lower for ev in evidence_list if len(ev.strip()) > 6):
        return "ignored_by_reader"

    if judge_call_fn is not None:
        prompt = EVIDENCE_SURVIVAL_PROMPT.format(
            question=question,
            evidence=evidence_joined,
            memory_context=memory_context[:24000],
        )
        verdict = judge_call_fn(prompt).strip().upper()
        if "PRESENT" in verdict:
            return "ignored_by_reader"
        return "lost_by_memory"

    return "lost_by_memory"
