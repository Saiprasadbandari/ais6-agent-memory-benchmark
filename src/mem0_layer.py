"""Task 3 (Owner: Majd Hammoud) - Mem0 Memory Layer (S4).

Implements three Mem0 settings on LongMemEval-S:
- M1: Default extraction, top_k = 5
- M2: Custom extraction instructions (keep dated events, preferences, and updated facts), top_k = 5
- M3: Default extraction, top_k = 10
Also tracks write-time ("lost: never stored") vs read-time ("ignored: stored but not used") errors.
"""

import time
from dataclasses import dataclass
from typing import Callable, List


DEFAULT_MEM0_EXTRACTION_PROMPT = (
    "Extract distinct factual statements and user preferences from the conversation segment. "
    "Return one concise fact per bullet line."
)

CUSTOM_M2_EXTRACTION_PROMPT = (
    "Extract all dated events, user preferences, numerical details, and updated/superseded facts "
    "from the conversation segment. Always preserve exact dates, timestamps, and when a preference "
    "changes, record both the previous and updated state. Return one fact per bullet line."
)


@dataclass
class Mem0RunResult:
    """Telemetry and retrieved context for a Mem0 configuration (M1, M2, or M3)."""

    setting: str
    top_k: int
    stored_memories: List[str]
    retrieved_memories: List[str]
    context_text: str
    build_tokens_used: int
    answer_prompt_tokens: int
    build_seconds: float
    search_seconds: float
    evidence_stored: int


def run_mem0_setting(
    setting: str,
    sessions_text: List[str],
    question: str,
    evidence_list: List[str],
    extract_fn: Callable[[str, str], List[str]],
    rank_fn: Callable[[str, List[str], int], List[str]],
    count_tokens_fn: Callable[[str], int],
) -> Mem0RunResult:
    """Build memory store under M1/M2/M3 and fetch top-k relevant facts for answering."""
    if setting == "M1":
        instruction = DEFAULT_MEM0_EXTRACTION_PROMPT
        top_k = 5
    elif setting == "M2":
        instruction = CUSTOM_M2_EXTRACTION_PROMPT
        top_k = 5
    elif setting == "M3":
        instruction = DEFAULT_MEM0_EXTRACTION_PROMPT
        top_k = 10
    else:
        raise ValueError(f"Unknown Mem0 setting: {setting}")

    t0 = time.perf_counter()
    stored_memories: List[str] = []
    build_tokens = 0

    for sess in sessions_text:
        build_tokens += count_tokens_fn(sess) + count_tokens_fn(instruction)
        extracted = extract_fn(sess, instruction)
        for fact in extracted:
            cleaned = fact.strip(" -•\t")
            if cleaned and cleaned not in stored_memories:
                stored_memories.append(cleaned)

    build_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    retrieved = rank_fn(question, stored_memories, top_k)
    search_seconds = time.perf_counter() - t1

    context_text = "\n".join(f"- {m}" for m in retrieved)
    store_joined = " ".join(stored_memories).lower()
    ev_stored = int(
        any(ev.lower() in store_joined for ev in evidence_list if len(ev.strip()) > 5)
    ) if evidence_list else 1

    return Mem0RunResult(
        setting=setting,
        top_k=top_k,
        stored_memories=stored_memories,
        retrieved_memories=retrieved,
        context_text=context_text,
        build_tokens_used=build_tokens,
        answer_prompt_tokens=count_tokens_fn(context_text),
        build_seconds=build_seconds,
        search_seconds=search_seconds,
        evidence_stored=ev_stored,
    )
