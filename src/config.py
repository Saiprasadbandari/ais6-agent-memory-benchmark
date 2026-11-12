"""Shared experimental configuration across all 3 modeling tasks (AIS 6)."""

from dataclasses import dataclass
from typing import Tuple

SEED: int = 42

# Shared Models
READER_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
JUDGE_MODEL_ID: str = "gpt-4o-mini"

# Native Context Window vs YaRN RoPE Scaling
NATIVE_CONTEXT_WINDOW: int = 32768
MAX_YARN_CONTEXT_WINDOW: int = 131072

# Shared Token Budgets
MAIN_TOKEN_BUDGET: int = 8192
TOKEN_BUDGETS: Tuple[int, ...] = (4096, 8192, 16384)

# Benchmark Settings
DATASET_HF_ID: str = "xiaowu0162/longmemeval-cleaned"
STRATIFIED_SUBSET_SIZE: int = 100
FULL_CONTEXT_SUBSET_SIZE: int = 30
HAND_VERIFICATION_SAMPLE_SIZE: int = 20


@dataclass(frozen=True)
class SummarizationPrompts:
    """Fixed summarization prompts reported across all Task 1 (S2) runs."""

    ONE_SHOT_ACON: str = (
        "You are a precision memory compressor for a long-running AI assistant.\n"
        "Compress the following multi-session conversation history into a structured summary "
        "within {budget_tokens} tokens. Preserve ALL exact dates, numbers, named entities, "
        "user preferences, and state updates.\n\n"
        "Structure your summary into four explicit sections:\n"
        "1. [USER PROFILE & PREFERENCES]: Enduring traits, likes, dislikes, constraints.\n"
        "2. [ENTITIES & KEY FACTS]: Explicit names, numbers, locations, and attributes.\n"
        "3. [CHRONOLOGICAL TIMELINE & UPDATES]: Dated sequence of events and superseded facts "
        "(clearly state both old and updated values when preferences/states change).\n"
        "4. [OPEN TASKS & CONTEXT]: Ongoing plans or unresolved items.\n\n"
        "Conversation History:\n{history}\n\nStructured Summary:"
    )

    RECURSIVE_UPDATE: str = (
        "You are maintaining a running memory summary for a long-term assistant.\n"
        "Merge the [PREVIOUS SUMMARY] with the [NEW CONVERSATION CHUNK] into an updated, "
        "consolidated summary within {budget_tokens} tokens.\n"
        "- Preserve all specific dates, numbers, entities, and user facts from both sources.\n"
        "- If the new chunk updates or contradicts an older fact, explicitly record the update "
        "with its timestamp.\n\n"
        "[PREVIOUS SUMMARY]:\n{previous_summary}\n\n"
        "[NEW CONVERSATION CHUNK]:\n{new_chunk}\n\n"
        "[UPDATED SUMMARY]:"
    )

    READER_QA: str = (
        "You are a helpful and accurate AI assistant. Answer the user's question using ONLY "
        "the provided conversation memory.\n"
        "- If the memory contains the answer, provide a concise, specific response.\n"
        "- If the question asks about something that was never mentioned or cannot be determined "
        "from the memory, explicitly abstain and state that the information is not available.\n\n"
        "Conversation Memory:\n{memory_context}\n\n"
        "Question (Asked at {question_date}): {question}\n"
        "Answer:"
    )
