"""Shared Qwen2.5-7B-Instruct (4-bit) reader & summarizer loader with YaRN support."""

from typing import Tuple
import torch

from src.config import MAX_YARN_CONTEXT_WINDOW, NATIVE_CONTEXT_WINDOW, READER_MODEL_ID


def load_qwen_reader_4bit(
    model_id: str = READER_MODEL_ID,
    enable_yarn_above_32k: bool = False,
) -> Tuple[object, object]:
    """Load 4-bit quantized Qwen2.5-7B-Instruct from Hugging Face.

    Switches on YaRN RoPE scaling (`factor=4.0`) only when inputs exceed the native 32k window.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    config = AutoConfig.from_pretrained(model_id)
    if enable_yarn_above_32k:
        config.rope_scaling = {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": NATIVE_CONTEXT_WINDOW,
        }
        config.max_position_embeddings = MAX_YARN_CONTEXT_WINDOW

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model
