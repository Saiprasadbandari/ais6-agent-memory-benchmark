"""Shared LongMemEval-S stratified loader and ground-truth evidence extractor."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.config import SEED, STRATIFIED_SUBSET_SIZE


def format_session_turns(session: List[Dict[str, Any]], session_date: str = "") -> str:
    """Format a single dialogue session into a readable string with date metadata."""
    header = f"--- Session ({session_date}) ---\n" if session_date else "--- Session ---\n"
    lines = [header]
    for turn in session:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_evidence_texts(item: Dict[str, Any]) -> List[str]:
    """Extract the ground-truth evidence turns/statements labeled in LongMemEval."""
    evidence_texts: List[str] = []
    haystack_sessions = item.get("haystack_sessions", [])
    answer_session_ids = set(item.get("answer_session_ids", []))
    haystack_session_ids = item.get("haystack_session_ids", [])

    for idx, session in enumerate(haystack_sessions):
        sess_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else None
        for turn in session:
            if turn.get("has_answer", False):
                evidence_texts.append(turn.get("content", "").strip())
            elif sess_id in answer_session_ids and not evidence_texts:
                # Fallback if turn-level flag is absent but session-level ID matches
                pass

    if not evidence_texts and "answer" in item:
        evidence_texts.append(str(item["answer"]))
    return evidence_texts


def stratified_sample_longmemeval(
    records: List[Dict[str, Any]],
    n_total: int = STRATIFIED_SUBSET_SIZE,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Sample a stratified subset balanced across LongMemEval question types."""
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        q_type = rec.get("question_type", "unknown")
        by_type.setdefault(q_type, []).append(rec)

    n_types = len(by_type)
    if n_types == 0:
        return []

    per_type = n_total // n_types
    remainder = n_total % n_types

    sampled: List[Dict[str, Any]] = []
    for idx, (q_type, group) in enumerate(sorted(by_type.items())):
        k = min(len(group), per_type + (1 if idx < remainder else 0))
        sampled.extend(rng.sample(group, k))

    return sampled


def save_subset(records: List[Dict[str, Any]], path: str | Path) -> None:
    """Save the stratified subset to disk so all 3 tasks run on identical questions."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
