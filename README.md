# Generative AI for Job Ad Enrichment

Multi-model benchmark project for structured extraction from job ads.

## Tech Stack

`Python` `PyTorch` `Transformers` `PEFT/QLoRA` `OpenAI API` `OpenRouter` `scikit-learn`

## What This Project Does

This project compares three solution families on 4 job-ad enrichment tasks:

- Rule-based baselines
- Proprietary LLMs (OpenAI + OpenRouter models)
- Llama-2 instruction fine-tuning

Tasks:

- Work arrangement classification (`onsite` / `hybrid` / `remote`)
- Salary extraction (`min-max-currency-frequency`)
- Seniority classification
- Job keyword/category classification

## Key Benchmark Highlights (Accuracy)

- **Work arrangement:** best `0.9495` (GPT-4o, Grok 3 mini) vs baseline `0.7475`
- **Salary extraction:** best `0.7954` (GPT-4o mini) vs baseline `0.0282`
- **Seniority:** best `0.6059` (Gemini 2.5 Flash) vs baseline `0.3251`
- **Keyword:** best `0.7600` (GPT-4.1) vs baseline `0.6800`

Results are computed from CSV outputs under:

- `Journeytothe West/MISC/baseline model results/`
- `Journeytothe West/MISC/Fine-tune LLMs Results/`

## Resume-Ready Bullets

### English

**Title line**

`Generative AI for Job Ad Enrichment | Python, PyTorch, Transformers, PEFT/QLoRA, OpenAI API, OpenRouter | GitHub`

**Bullet options**

- Built and benchmarked rule-based baselines, proprietary LLMs, and Llama-2 fine-tuning for 4 job-ad enrichment tasks (salary, seniority, keyword, work arrangement).
- Implemented structured extraction and evaluation pipelines, improving work-arrangement accuracy from `0.75` baseline to `0.95` with top-performing LLMs.
- Designed prompt-based and fine-tuned model workflows for noisy real-world job-ad text and compared multi-model trade-offs across task difficulty.

### 中文

**项目标题**

`招聘广告生成式 AI 增强 | Python, PyTorch, Transformers, PEFT/QLoRA, OpenAI API, OpenRouter | GitHub`

**描述要点**

- 搭建并对比规则基线、商业大模型 API 与 Llama-2 微调三类方案，覆盖薪资、职级、岗位关键词与办公方式 4 个任务。
- 实现结构化抽取与评测流水线，在办公方式识别任务中将准确率从基线约 `0.75` 提升到约 `0.95`。
- 针对真实招聘文本噪声与标签差异，完成多模型效果对比并分析任务难度与模型选择的权衡。

## Repository Layout

- `Journeytothe West/Code/Baseline Models/`: rule-based baseline scripts
- `Journeytothe West/Code/Proprietary Models/`: OpenAI/OpenRouter evaluation scripts
- `Journeytothe West/Code/fine_tuning_LLMs/`: Llama-2 fine-tuning scripts and notebooks
- `Journeytothe West/MISC/Input file dataset/`: input CSV datasets
- `Journeytothe West/MISC/baseline model results/`: baseline outputs
- `Journeytothe West/MISC/Fine-tune LLMs Results/`: multi-model outputs

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "Journeytothe West/MISC/requirements.txt"
pip install pandas scikit-learn tqdm openai chardet datasets
```

Run a baseline example:

```bash
python "Journeytothe West/Code/Baseline Models/baseline_work_ar.py"
```

For API evaluations, set your API key + model name in:

- `Journeytothe West/Code/Proprietary Models/OpenAI.py`
- `Journeytothe West/Code/Proprietary Models/OpenRouter.py`
