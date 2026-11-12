"""Task 1 (Owner: Sai Prasad Bandari) - Summarization as Memory (S2).

Implements:
- Reference: Truncation (keep latest messages up to token budget)
- Variant (a): One-shot summarization (ACON-style structured state summary; chunked & merged if >32k)
- Variant (b): Recursive summarization (new_summary = summarize(previous_summary + next_chunk))
- Variant (c): Summary + recent messages (summary of older part + latest messages verbatim)
"""

import time
from dataclasses import dataclass
from typing import Callable, List

from src.config import MAIN_TOKEN_BUDGET, NATIVE_CONTEXT_WINDOW, SummarizationPrompts


@dataclass
class MemoryBuildResult:
    """Container for constructed memory context and token/latency telemetry."""

    variant: str
    context_text: str
    history_tokens: int
    memory_tokens: int
    build_tokens_used: int
    build_seconds: float
    num_model_calls: int

    @property
    def compression_ratio(self) -> float:
        """Compute history_tokens / memory_tokens."""
        return self.history_tokens / max(1, self.memory_tokens)


def truncate_latest_tokens(
    sessions_text: List[str],
    count_tokens_fn: Callable[[str], int],
    budget_tokens: int = MAIN_TOKEN_BUDGET,
) -> MemoryBuildResult:
    """Reference S1: Keep the latest messages/turns up to the target token budget."""
    t0 = time.perf_counter()
    full_text = "\n\n".join(sessions_text)
    history_tokens = count_tokens_fn(full_text)

    kept_chunks: List[str] = []
    running_tokens = 0
    for session_str in reversed(sessions_text):
        t_len = count_tokens_fn(session_str)
        if running_tokens + t_len <= budget_tokens:
            kept_chunks.append(session_str)
            running_tokens += t_len
        else:
            remaining = budget_tokens - running_tokens
            if remaining > 64:
                words = session_str.split()
                approx_words = max(1, int(len(words) * (remaining / max(1, t_len))))
                partial = " ".join(words[-approx_words:])
                kept_chunks.append(partial)
                running_tokens += count_tokens_fn(partial)
            break

    kept_chunks.reverse()
    final_context = "\n\n".join(kept_chunks)
    elapsed = time.perf_counter() - t0
    return MemoryBuildResult(
        variant="truncation_ref",
        context_text=final_context,
        history_tokens=history_tokens,
        memory_tokens=count_tokens_fn(final_context),
        build_tokens_used=0,
        build_seconds=elapsed,
        num_model_calls=0,
    )


def summarize_one_shot(
    sessions_text: List[str],
    generate_fn: Callable[[str, int], str],
    count_tokens_fn: Callable[[str], int],
    budget_tokens: int = MAIN_TOKEN_BUDGET,
    max_window_tokens: int = NATIVE_CONTEXT_WINDOW - 2048,
) -> MemoryBuildResult:
    """Variant (a): One-shot summary of full history (chunked & merged if exceeding native window)."""
    t0 = time.perf_counter()
    full_text = "\n\n".join(sessions_text)
    history_tokens = count_tokens_fn(full_text)
    build_tokens = 0
    calls = 0

    if history_tokens <= max_window_tokens:
        prompt = SummarizationPrompts.ONE_SHOT_ACON.format(
            budget_tokens=budget_tokens,
            history=full_text,
        )
        build_tokens += count_tokens_fn(prompt)
        summary = generate_fn(prompt, budget_tokens)
        calls += 1
    else:
        # Chunk into windows that fit native 32k context, summarize each, then merge
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0
        for sess in sessions_text:
            s_tok = count_tokens_fn(sess)
            if current_tokens + s_tok > max_window_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [sess]
                current_tokens = s_tok
            else:
                current_chunk.append(sess)
                current_tokens += s_tok
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        sub_budget = max(512, budget_tokens // max(1, len(chunks)))
        partial_summaries: List[str] = []
        for ch in chunks:
            prompt = SummarizationPrompts.ONE_SHOT_ACON.format(
                budget_tokens=sub_budget,
                history=ch,
            )
            build_tokens += count_tokens_fn(prompt)
            partial_summaries.append(generate_fn(prompt, sub_budget))
            calls += 1

        merged_input = "\n\n".join(partial_summaries)
        merge_prompt = SummarizationPrompts.ONE_SHOT_ACON.format(
            budget_tokens=budget_tokens,
            history=merged_input,
        )
        build_tokens += count_tokens_fn(merge_prompt)
        summary = generate_fn(merge_prompt, budget_tokens)
        calls += 1

    elapsed = time.perf_counter() - t0
    return MemoryBuildResult(
        variant="s2_a_one_shot",
        context_text=summary,
        history_tokens=history_tokens,
        memory_tokens=count_tokens_fn(summary),
        build_tokens_used=build_tokens,
        build_seconds=elapsed,
        num_model_calls=calls,
    )


def summarize_recursive(
    sessions_text: List[str],
    generate_fn: Callable[[str, int], str],
    count_tokens_fn: Callable[[str], int],
    budget_tokens: int = MAIN_TOKEN_BUDGET,
    chunk_token_size: int = 16384,
) -> MemoryBuildResult:
    """Variant (b): Recursive summary: new_summary = summarize(previous_summary + next_chunk)."""
    t0 = time.perf_counter()
    full_text = "\n\n".join(sessions_text)
    history_tokens = count_tokens_fn(full_text)

    # Group sessions into manageable chunks
    chunks: List[str] = []
    curr: List[str] = []
    curr_tok = 0
    for sess in sessions_text:
        s_tok = count_tokens_fn(sess)
        if curr_tok + s_tok > chunk_token_size and curr:
            chunks.append("\n\n".join(curr))
            curr = [sess]
            curr_tok = s_tok
        else:
            curr.append(sess)
            curr_tok += s_tok
    if curr:
        chunks.append("\n\n".join(curr))

    running_summary = "No prior history."
    build_tokens = 0
    calls = 0

    for chunk in chunks:
        prompt = SummarizationPrompts.RECURSIVE_UPDATE.format(
            budget_tokens=budget_tokens,
            previous_summary=running_summary,
            new_chunk=chunk,
        )
        build_tokens += count_tokens_fn(prompt)
        running_summary = generate_fn(prompt, budget_tokens)
        calls += 1

    elapsed = time.perf_counter() - t0
    return MemoryBuildResult(
        variant="s2_b_recursive",
        context_text=running_summary,
        history_tokens=history_tokens,
        memory_tokens=count_tokens_fn(running_summary),
        build_tokens_used=build_tokens,
        build_seconds=elapsed,
        num_model_calls=calls,
    )


def summarize_hybrid_with_recent(
    sessions_text: List[str],
    generate_fn: Callable[[str, int], str],
    count_tokens_fn: Callable[[str], int],
    budget_tokens: int = MAIN_TOKEN_BUDGET,
    recent_ratio: float = 0.5,
) -> MemoryBuildResult:
    """Variant (c): Summary of older sessions + verbatim recent messages at fixed total budget."""
    t0 = time.perf_counter()
    full_text = "\n\n".join(sessions_text)
    history_tokens = count_tokens_fn(full_text)

    recent_budget = int(budget_tokens * recent_ratio)
    summary_budget = budget_tokens - recent_budget

    # Partition sessions from the end into verbatim recent vs older history
    recent_sessions: List[str] = []
    recent_tok = 0
    split_idx = len(sessions_text)

    for idx in range(len(sessions_text) - 1, -1, -1):
        s_tok = count_tokens_fn(sessions_text[idx])
        if recent_tok + s_tok <= recent_budget:
            recent_sessions.append(sessions_text[idx])
            recent_tok += s_tok
            split_idx = idx
        else:
            break

    recent_sessions.reverse()
    older_sessions = sessions_text[:split_idx]

    if older_sessions:
        older_res = summarize_recursive(
            older_sessions,
            generate_fn=generate_fn,
            count_tokens_fn=count_tokens_fn,
            budget_tokens=summary_budget,
        )
        summary_part = older_res.context_text
        build_tokens = older_res.build_tokens_used
        calls = older_res.num_model_calls
    else:
        summary_part = ""
        build_tokens = 0
        calls = 0

    recent_part = "\n\n".join(recent_sessions)
    combined = (
        f"[SUMMARY OF EARLIER SESSIONS]:\n{summary_part}\n\n"
        f"[RECENT VERBATIM MESSAGES]:\n{recent_part}"
    ).strip()

    elapsed = time.perf_counter() - t0
    return MemoryBuildResult(
        variant="s2_c_summary_plus_recent",
        context_text=combined,
        history_tokens=history_tokens,
        memory_tokens=count_tokens_fn(combined),
        build_tokens_used=build_tokens,
        build_seconds=elapsed,
        num_model_calls=calls,
    )
