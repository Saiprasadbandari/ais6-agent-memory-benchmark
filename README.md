# How Should an LLM Agent Remember?
**A Comparative Study of Memory Management Strategies for Long-Running LLM Agents (Action Learning — Team AIS 6)**

---

## Overview & Research Question
An LLM agent operating over many sessions must fit ~115k-token conversation histories into a constrained token budget. This repository implements a controlled, reproducible benchmark comparing **five memory management strategies ($S_0$–$S_4$)** on **LongMemEval-S (ICLR 2025)** using a shared open reader model (**Qwen2.5-7B-Instruct, 4-bit**) and a standardized LLM judge (**GPT-4o-mini**).

> **Core Project Question:** *Which memory management strategy lets an LLM agent answer correctly over very long interactions at the lowest cost in tokens and time, and when the agent fails, was the required evidence **lost by the memory** or **ignored by the reader model**?*

---

## Team & Task Allocation (AIS 6)

| Task | Owner | Strategy & Scope | Core Papers | Notebook |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Sai Prasad Bandari** | **Summarization as Memory ($S_2$)**<br>• `(a)` One-shot (ACON structured compression)<br>• `(b)` Recursive (`summarize(prev + chunk)`)<br>• `(c)` Summary + recent verbatim messages<br>• *Ref:* Truncation ($S_1$) at 8k token budget | `#3` ACON<br>`#4` ReSum<br>`#18` Recursive Summarization<br>`#1` LongMemEval<br>`#15` Lost in the Middle | [`notebooks/SaiPrasadBandari_Summarization_Memory.ipynb`](notebooks/SaiPrasadBandari_Summarization_Memory.ipynb) |
| **Task 2** | **Praveen Kumar Koppula** | **Truncation, Full Context & Retrieval ($S_0, S_1, S_3$)**<br>• **Truncation ($S_1$):** 4k, 8k, 16k tokens<br>• **Retrieval ($S_3$):** `all-MiniLM-L6-v2` + FAISS (turn, session, segment $\times k \in \{3,5,10\}$)<br>• **Full Context ($S_0$):** YaRN $>32\text{k}$ reference | `#5` SeCom<br>`#8` Retrieval vs Utilization<br>`#12` Self-Route<br>`#16` RULER<br>`#25` SBERT / `#26` FAISS | [`notebooks/PraveenKumarKoppula_Truncation_And_Retrieval.ipynb`](notebooks/PraveenKumarKoppula_Truncation_And_Retrieval.ipynb) |
| **Task 3** | **Majd Hammoud** | **Mem0 Memory Layer ($S_4$)**<br>• **M1:** Default extraction, $\text{top-}k=5$<br>• **M2:** Custom dated-event & preference instructions, $\text{top-}k=5$<br>• **M3:** Default extraction, $\text{top-}k=10$ | `#6` Mem0<br>`#19` LoCoMo<br>`#20` Zep<br>`#9, #10` Memory Surveys<br>`#1` LongMemEval | [`notebooks/MajdHammoud_Mem0_Memory_Layer.ipynb`](notebooks/MajdHammoud_Mem0_Memory_Layer.ipynb) |

**Faculty Advisor:** Alaa Bakhti · **Program Leads:** Rosemary Obi, Bill Manos

---

## Shared Experimental Protocol
All three tasks run independently from Day 1 on top of a unified foundation (`src/`):
- **Reader & Summarizer Model:** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 quantized via `bitsandbytes`, greedy decoding `do_sample=False`, `SEED = 42`, YaRN RoPE scaling enabled only when input exceeds native 32k context).
- **Benchmark:** `LongMemEval-S` (ICLR 2025) — 100-question subset stratified across all five memory capabilities (*Information Extraction*, *Multi-Session Reasoning*, *Temporal Reasoning*, *Knowledge Updates*, and *Abstention*), paired with session- and turn-level ground-truth evidence labels.
- **Evaluation Judge:** `gpt-4o-mini` following the official LongMemEval evaluation rubric.
- **Statistical Rigor:** 95% bootstrap confidence intervals ($B=1000$) and paired **McNemar tests** between strategy variants on identical questions.
- **Deeper Diagnostic ("Lost vs. Ignored"):** Every incorrect answer is audited against the dataset's ground-truth evidence turns:
  - **Lost by Memory:** Required evidence is absent from the compressed summary / retrieved prompt / Mem0 store.
  - **Ignored by Reader:** Required evidence survived in the prompt, but the reader failed to utilize it.

---

## Repository Structure

```text
ais6-agent-memory-benchmark/
├── notebooks/
│   ├── SaiPrasadBandari_Summarization_Memory.ipynb
│   ├── PraveenKumarKoppula_Truncation_And_Retrieval.ipynb
│   └── MajdHammoud_Mem0_Memory_Layer.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py           # Shared constants, seed=42, token budgets (4k/8k/16k), model settings
│   ├── data_loader.py      # Stratified 100-question LongMemEval-S loader & evidence parser
│   ├── summarization.py    # Task 1 (S2): One-shot (ACON), Recursive (ReSum), Summary + Recent
│   ├── judge.py            # Shared GPT-4o-mini judge & Lost-vs-Ignored evidence survival verifier
│   └── metrics.py          # 95% Bootstrap CI, paired McNemar test, token/latency accounting
├── data/
│   ├── raw/                # Raw LongMemEval-S dataset files
│   └── processed/          # Stratified 100-question subset & cached evidence spans
├── results/
│   ├── summaries/          # Pre-computed & cached summaries (Task 1)
│   ├── metrics/            # Per-question evaluation logs (.csv / .json)
│   └── figures/            # Accuracy-vs-cost curves & Lost-vs-Ignored breakdown plots
├── docs/                   # Modeling Tasks definition PDFs
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Saiprasadbandari/ais6-agent-memory-benchmark.git
cd ais6-agent-memory-benchmark

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set OpenAI API key for the shared GPT-4o-mini Judge
export OPENAI_API_KEY="your-api-key"
```
