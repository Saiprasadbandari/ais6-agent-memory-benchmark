"""AIS 6 Interactive Memory Workbench (MVP).

Allows users and developers to:
1. Inspect long multi-session conversations & switch live between S1 (Truncation),
   S2 (One-Shot ACON, Recursive ReSum, Summary + Recent), S3 (FAISS Retrieval), and S4 (Mem0).
2. View live token budgets, build vs. query latency, compression ratios, and Evidence Survival.
3. Audit 'Lost by Memory' vs. 'Ignored by Reader' diagnostics per question type and receive
   an automated strategy recommendation based on token/latency constraints.
"""

import json
from pathlib import Path
import pandas as pd

from src.summarization import (
    truncate_latest_tokens,
    summarize_one_shot,
    summarize_recursive,
    summarize_hybrid_with_recent,
)


def recommend_memory_strategy(
    question_type: str,
    max_prompt_tokens: int = 8192,
    prioritize_low_answer_latency: bool = False,
) -> dict:
    """Return an evidence-backed strategy recommendation from AIS 6 benchmark results."""
    if prioritize_low_answer_latency and max_prompt_tokens <= 1024:
        return {
            "recommended": "S4: Mem0 Custom Extraction (M2, top_k=5)",
            "expected_accuracy_pct": 72.0,
            "avg_prompt_tokens": 437,
            "rationale": (
                "Reduces answer-time tokens by 99.6% (~437 tokens) with 0.82s answer latency "
                "while preserving dated events and knowledge updates (83% evidence coverage)."
            ),
        }
    if question_type in ("multi-session-reasoning", "temporal-reasoning", "knowledge-update"):
        if max_prompt_tokens >= 7500:
            return {
                "recommended": "S2(c): Summary + Recent Messages (8k budget)",
                "expected_accuracy_pct": 76.0,
                "avg_prompt_tokens": 7810,
                "rationale": (
                    "Combines recursive global state compression with verbatim recent turns, "
                    "achieving 87% evidence survival and +47 accuracy points over truncation."
                ),
            }
        return {
            "recommended": "S3: Dense Retrieval — Topic Segment (top_k=5)",
            "expected_accuracy_pct": 77.0,
            "avg_prompt_tokens": 2848,
            "rationale": (
                "Captures local multi-turn context around each retrieved fact at 1/4th the "
                "token budget of 8k summaries."
            ),
        }
    return {
        "recommended": "S3: Dense Retrieval — Topic Segment (top_k=5)",
        "expected_accuracy_pct": 77.0,
        "avg_prompt_tokens": 2848,
        "rationale": "Highest accuracy-per-token across single-session detail and abstention queries.",
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    t1 = pd.read_csv(base / "results/metrics/task1_summary.csv")
    t2 = pd.read_csv(base / "results/metrics/task2_summary.csv")
    t3 = pd.read_csv(base / "results/metrics/task3_summary.csv")

    print("=====================================================================")
    print("  AIS 6 — Long-Running LLM Agent Memory Benchmark & Strategy Advisor ")
    print("=====================================================================\n")
    print("--- Task 1: Summarization as Memory (S2) ---")
    print(t1[["Variant", "Accuracy (%)", "95% CI", "Evidence Survival (%)", "Lost Share (%)", "Ignored Share (%)"]].to_string(index=False))
    print("\n--- Task 2: Truncation, Full Context & Retrieval (S0, S1, S3) ---")
    print(t2[["Strategy", "Accuracy (%)", "95% CI", "Turn Recall (%)", "Prompt Tokens", "Lost (%)", "Ignored (%)"]].to_string(index=False))
    print("\n--- Task 3: Mem0 Memory Layer (S4) ---")
    print(t3[["Setting", "Accuracy (%)", "95% CI", "Evidence Coverage (%)", "Answer Tokens", "Lost at Write (%)", "Ignored at Read (%)"]].to_string(index=False))

    print("\n--- Automated Strategy Advisor ---")
    for q_t in ["single-session-detail", "multi-session-reasoning", "knowledge-update"]:
        rec = recommend_memory_strategy(q_t, max_prompt_tokens=8192)
        print(f"[{q_t}] -> {rec['recommended']} ({rec['expected_accuracy_pct']}% acc, ~{rec['avg_prompt_tokens']} prompt tokens)")
