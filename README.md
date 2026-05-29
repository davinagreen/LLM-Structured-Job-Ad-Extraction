# LLM Structured Job Ad Extraction

An LLM-centric project for extracting structured information from noisy job advertisements, with side-by-side benchmarking against rule-based baselines.

## Project Snapshot

- **Goal:** convert raw job-ad text into structured fields for downstream search, matching, and analytics
- **Task types (4):**
  - work arrangement (`onsite` / `hybrid` / `remote`)
  - salary extraction (`min-max-currency-frequency`)
  - seniority classification
  - keyword/category classification
- **Approaches compared:**
  - rule-based baseline models
  - proprietary LLMs (OpenAI + OpenRouter)
  - Llama-2 fine-tuning pipeline (QLoRA/PEFT)

## Tech Stack

`Python` `PyTorch` `Transformers` `PEFT/QLoRA` `OpenAI API` `OpenRouter` `scikit-learn` `pandas`

## Benchmark Highlights (Accuracy)

- **Work arrangement:** `0.9495` (GPT-4o / Grok 3 mini) vs baseline `0.7475`
- **Salary extraction:** `0.7954` (GPT-4o mini) vs baseline `0.0282`
- **Seniority:** `0.6059` (Gemini 2.5 Flash) vs baseline `0.3251`
- **Keyword:** `0.7600` (GPT-4.1) vs baseline `0.6800`

Raw outputs are available in:
- `Journeytothe West/MISC/baseline model results/`
- `Journeytothe West/MISC/Fine-tune LLMs Results/`

## Repository Structure

- `Journeytothe West/Code/Baseline Models/`  
  Rule-based scripts for the 4 tasks
- `Journeytothe West/Code/Proprietary Models/`  
  Evaluation scripts for OpenAI/OpenRouter models
- `Journeytothe West/Code/fine_tuning_LLMs/`  
  Llama-2 fine-tuning scripts and notebooks
- `Journeytothe West/MISC/Input file dataset/`  
  Input CSV datasets
- `Journeytothe West/MISC/Fine-tune LLMs Results/`  
  Multi-model inference outputs

## Quick Start

### 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "Journeytothe West/MISC/requirements.txt"
pip install pandas scikit-learn tqdm openai chardet datasets
```

### 2) Run baseline models

```bash
python "Journeytothe West/Code/Baseline Models/baseline_work_ar.py"
python "Journeytothe West/Code/Baseline Models/baseline_salary.py"
python "Journeytothe West/Code/Baseline Models/baseline_seniority.py"
python "Journeytothe West/Code/Baseline Models/baseline_keywords.py"
```

### 3) Run proprietary LLM evaluations

Set your API key and target model in:
- `Journeytothe West/Code/Proprietary Models/OpenAI.py`
- `Journeytothe West/Code/Proprietary Models/OpenRouter.py`

Then execute:

```bash
python "Journeytothe West/Code/Proprietary Models/OpenAI.py"
# or
python "Journeytothe West/Code/Proprietary Models/OpenRouter.py"
```

### 4) Run Llama-2 fine-tuning

See:
- `Journeytothe West/Code/fine_tuning_LLMs/train.py`
- `Journeytothe West/Code/fine_tuning_LLMs/myllama.py`
- notebooks under `Journeytothe West/Code/fine_tuning_LLMs/`

## Resume Version (Copy-Paste)

**Title**

`LLM Structured Job Ad Extraction | Python, PyTorch, Transformers, PEFT/QLoRA, OpenAI API, OpenRouter | GitHub`

**Bullet**

Built and benchmarked rule-based baselines, proprietary LLM APIs, and Llama-2 fine-tuning for 4 job-ad extraction tasks (salary, seniority, keyword, work arrangement), with best-task accuracy reaching 0.95 and clear gains over baseline methods.

## Notes

- This repository is an academic project with experiment artifacts.
- Some scripts are exploratory and may require path adjustments before rerun.
- API-based scripts require your own credentials.
