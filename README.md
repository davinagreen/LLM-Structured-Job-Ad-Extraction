# Generative AI for Job Ad Enrichment

This project benchmarks multiple approaches for extracting structured information from job advertisements.

## Project Scope

We evaluate 4 tasks:

- Work arrangement classification (`onsite`, `hybrid`, `remote`)
- Salary extraction (`min-max-currency-frequency`)
- Seniority classification
- Job keyword/category classification

Approaches compared:

- Rule-based baselines
- Proprietary LLM APIs (OpenAI and OpenRouter models)
- Llama-2 instruction fine-tuning (QLoRA / PEFT)

## Repository Layout

- `Journeytothe West/Code/Baseline Models/`
  - Rule-based baseline scripts for all 4 tasks
- `Journeytothe West/Code/Proprietary Models/`
  - API evaluation scripts for OpenAI and OpenRouter models
- `Journeytothe West/Code/fine_tuning_LLMs/`
  - Llama-2 fine-tuning scripts and notebooks
- `Journeytothe West/MISC/Input file dataset/`
  - Input CSV datasets
- `Journeytothe West/MISC/baseline model results/`
  - Baseline prediction outputs
- `Journeytothe West/MISC/Fine-tune LLMs Results/`
  - Multi-model evaluation outputs

## Environment

Minimum:

- Python 3.10+

Suggested install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "Journeytothe West/MISC/requirements.txt"
pip install pandas scikit-learn tqdm openai chardet datasets
```

## How to Run

### 1) Baseline models

Run scripts inside:

`Journeytothe West/Code/Baseline Models/`

Example:

```bash
python "Journeytothe West/Code/Baseline Models/baseline_work_ar.py"
```

### 2) Proprietary model evaluation

Edit API key and model name in:

- `Journeytothe West/Code/Proprietary Models/OpenAI.py`
- `Journeytothe West/Code/Proprietary Models/OpenRouter.py`

Then run:

```bash
python "Journeytothe West/Code/Proprietary Models/OpenAI.py"
```

### 3) Llama-2 fine-tuning

See:

- `Journeytothe West/Code/fine_tuning_LLMs/train.py`
- `Journeytothe West/Code/fine_tuning_LLMs/myllama.py`

## Notes

- This repository contains coursework artifacts and experiment outputs.
- API-based scripts require your own API credentials.
- Some notebooks are exploratory and may need path updates before rerunning.
