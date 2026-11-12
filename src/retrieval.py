"""Task 2 (Owner: Praveen Kumar Koppula) - Truncation, Full Context and Retrieval (S0, S1, S3).

Implements:
- Truncation (S1): Keep latest 4k, 8k, or 16k tokens
- Retrieval (S3): all-MiniLM-L6-v2 embeddings + FAISS index across units:
    * single_turn
    * whole_session
    * topic_segment
  with top_k in {3, 5, 10}, capped to match truncation token budgets
- Full-context reference (S0): Whole history (with YaRN RoPE scaling above 32k tokens)
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple
import numpy as np

from src.config import MAIN_TOKEN_BUDGET
from src.summarization import truncate_latest_tokens

RetrievalUnit = Literal["single_turn", "whole_session", "topic_segment"]


@dataclass
class RetrievalResult:
    """Container for retrieved context and retrieval-quality telemetry."""

    unit: RetrievalUnit
    top_k: int
    budget_tokens: int
    context_text: str
    retrieved_indices: List[int]
    evidence_recall_turn: int
    evidence_recall_session: int
    prompt_tokens: int
    index_seconds: float
    search_seconds: float


def build_retrieval_units(
    haystack_sessions: List[List[Dict[str, Any]]],
    session_dates: List[str] | None = None,
    unit: RetrievalUnit = "single_turn",
    segment_window: int = 4,
) -> List[Dict[str, Any]]:
    """Chunk a multi-session conversation history into the specified retrieval unit."""
    chunks: List[Dict[str, Any]] = []
    dates = session_dates or [""] * len(haystack_sessions)

    for s_idx, session in enumerate(haystack_sessions):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        if unit == "single_turn":
            for t_idx, turn in enumerate(session):
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "").strip()
                has_ans = bool(turn.get("has_answer", False))
                text = f"[{s_date}] {role}: {content}" if s_date else f"{role}: {content}"
                chunks.append({
                    "text": text,
                    "session_idx": s_idx,
                    "turn_idx": t_idx,
                    "has_answer": has_ans,
                })
        elif unit == "whole_session":
            lines = [f"--- Session ({s_date}) ---" if s_date else "--- Session ---"]
            has_ans = False
            for turn in session:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "").strip()
                if turn.get("has_answer", False):
                    has_ans = True
                lines.append(f"{role}: {content}")
            chunks.append({
                "text": "\n".join(lines),
                "session_idx": s_idx,
                "turn_idx": -1,
                "has_answer": has_ans,
            })
        elif unit == "topic_segment":
            for start in range(0, len(session), segment_window):
                window = session[start : start + segment_window]
                lines = [f"[{s_date} Segment]" if s_date else "[Segment]"]
                has_ans = False
                for turn in window:
                    role = turn.get("role", "user").capitalize()
                    content = turn.get("content", "").strip()
                    if turn.get("has_answer", False):
                        has_ans = True
                    lines.append(f"{role}: {content}")
                chunks.append({
                    "text": "\n".join(lines),
                    "session_idx": s_idx,
                    "turn_idx": start,
                    "has_answer": has_ans,
                })
    return chunks


def retrieve_top_k_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    embed_fn: Callable[[List[str]], np.ndarray],
    count_tokens_fn: Callable[[str], int],
    unit: RetrievalUnit = "single_turn",
    top_k: int = 5,
    budget_tokens: int = MAIN_TOKEN_BUDGET,
) -> RetrievalResult:
    """Embed chunks, search top-k by cosine similarity (FAISS/dot-product), and cap at budget."""
    t0 = time.perf_counter()
    texts = [c["text"] for c in chunks]
    if not texts:
        return RetrievalResult(unit, top_k, budget_tokens, "", [], 0, 0, 0, 0.0, 0.0)

    doc_embs = embed_fn(texts)
    doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12
    doc_embs = doc_embs / doc_norms
    index_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    q_emb = embed_fn([question])
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    try:
        import faiss  # type: ignore
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs.astype(np.float32))
        k_eff = min(top_k, len(texts))
        _, indices_arr = index.search(q_emb.astype(np.float32), k_eff)
        top_indices = [int(i) for i in indices_arr[0] if i >= 0]
    except ImportError:
        scores = np.dot(doc_embs, q_emb[0])
        k_eff = min(top_k, len(texts))
        top_indices = [int(i) for i in np.argsort(-scores)[:k_eff]]

    search_seconds = time.perf_counter() - t1

    # Cap retrieved text to the target token budget
    selected_texts: List[str] = []
    selected_indices: List[int] = []
    running_tokens = 0
    for idx in top_indices:
        chunk_text = chunks[idx]["text"]
        t_count = count_tokens_fn(chunk_text)
        if running_tokens + t_count <= budget_tokens:
            selected_texts.append(chunk_text)
            selected_indices.append(idx)
            running_tokens += t_count
        else:
            break

    evidence_sessions = {c["session_idx"] for c in chunks if c["has_answer"]}
    retrieved_sessions = {chunks[i]["session_idx"] for i in selected_indices}
    recall_turn = int(any(chunks[i]["has_answer"] for i in selected_indices))
    recall_session = int(bool(evidence_sessions & retrieved_sessions)) if evidence_sessions else recall_turn

    context_text = "\n\n".join(selected_texts)
    return RetrievalResult(
        unit=unit,
        top_k=top_k,
        budget_tokens=budget_tokens,
        context_text=context_text,
        retrieved_indices=selected_indices,
        evidence_recall_turn=recall_turn,
        evidence_recall_session=recall_session,
        prompt_tokens=count_tokens_fn(context_text),
        index_seconds=index_seconds,
        search_seconds=search_seconds,
    )
