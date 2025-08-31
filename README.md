# LLM Political Bias & Personality — Data Collection & Analysis

This repository contains a small, reproducible toolkit to **collect** model responses for two widely used psychometrics (Big Five Personality Test/ BFI‑50 and  Political Compass Test) and to **analyze** those responses across model variants and role/persona prompts. It is designed to be portable (single‑file scripts), transparent, and friendly to open data workflows.

In our published case study, we focused on three OpenAI models (GPT-3.5, GPT-4, GPT-4o).  
However, the framework is not limited to these models — it can be applied to **any LLM** as long as model-specific adjustments (API client, endpoint, formatting) are made.

> **What you can do with this repo**
>
> 1) Collect answers from Large Language Models (LLMs) (e.g., GPT‑3.5 / GPT‑4 / GPT‑4o) for Big Five and Political Compass questionnaires.  
> 2) Run statistical analysis (means, SDs, pairwise Welch’s t‑tests).  
> 3) Generate clean CSV/XLSX outputs and publication‑ready plots.  
> 4) Swap in your own models or your own datasets.

---

## Repository structure

```
.
├── collect_bigfive_responses.py              # Big Five Personality Test (BFI‑50) data collection
├── collect_political_compass_responses.py    # Political Compass Test data collection
├── collect_bigfive_compass_responses.py      # Script to collect sequential Big Five + Compass in sequential runs
├── bigfive_analysis.py                       # Big Five analysis / stats / plots
├── political_compass_analysis.py             # Political Compass analysis / stats / plots
├── sequential_analysis.py                    # Analysis for sequential runs (order effects: Big Five ↔ Compass or  Compass ↔ Big Five  )
├── requirements.txt                          # Pinned dependencies for all scripts
└── data/                                     # Compiled responses (.xlsx) per model
```

### Data folder (optional but recommended)

Place compiled response files here (or https://zenodo.org/records/13842621):

```
data/
├── bigfive_gpt3_5.xlsx
├── bigfive_gpt4.xlsx
├── bigfive_gpt_4o.xlsx
├── political_compass_gpt3_5.xlsx
├── political_compass_gpt4.xlsx
├── political_compass_gpt-4o.xlsx
├── sequential_tests_gpt3_5.xlsx
└── ...
```

Each workbook uses **one sheet per role/persona** (e.g., `Default`, `Left-Authoritarian`, `Right-Authoritarian`, `Left-Libertarian`, `Right-Libertarian`).  
- **Big Five (BFI-50)**  
  - Columns: `X1` ... `X50`  
  - Each cell contains one of five Likert responses:  
    *Strongly Agree, Agree, Neutral, Disagree, Strongly Disagree*

- **Political Compass Test (62 items)**  
  - Columns: `X1` ... `X62`  
  - Each cell contains one of four Likert responses:  
    *Strongly Agree, Agree, Disagree, Strongly Disagree*

---

## Installation

Tested with **Python 3.9–3.11**.

## Collecting data

### 1) Big Five Personality Test

File: `collect_bigfive_responses.py`  
- Reads `OPENAI_API_KEY` from the environment.  
- Edit these constants at the bottom of the file before running:
  - `selected_profile` (e.g., `"default"`, `"left-authoritarian"`, `"right-libertarian"`)
  - `num_surveys` (number of full 50‑item runs)
  - `model` (e.g., `"gpt-4o"`)


### 2) Political Compass Test

File: `collect_political_compass_responses.py`  
- Reads `OPENAI_API_KEY` from the environment.  
- Edit these constants at the bottom of the file before running:
  - `selected_profile` (e.g., `"default"`, `"left-authoritarian"`, `"right-libertarian"`)
  - `num_surveys` (number of full 62‑item runs)
  - `model` (e.g., `"gpt-4o"`)

### 3) Sequential (Big Five → Political Compass or Political Compass → Big Five)

File: `collect_bigfive_compass_responses.py`  
This script lets you pick order, persona, model, and number of runs from the command line.

## Analyzing data

All analysis scripts accept **local file paths or direct URLs** (e.g., Zenodo) for input Excel files.

### Big Five Personality Analysis

File: `bigfive_analysis.py`

- Inputs: one or more workbooks (per model).  
- Each workbook: sheets = roles; columns `X1..X50` = Likert answers (words or 1–5).  
- Computes OCEAN % scores with reverse‑keyed items.  
- Outputs:
  - `combined_bigfive_scores.csv` (per‑response OCEAN %)
  - `summary_bigfive_stats.csv` (means/SDs by model×role×trait)
  - `welch_ocean_results.csv` / `.xlsx` (pairwise Welch’s t (potentially with multiplicity adjustment) + Hedges’ g & 95% CI)
  - `OCEAN_BigFive_GPT.pdf` (bar chart for default role)

### Political Compass Analysis

File: `political_compass_analysis.py`

- Inputs: one or more workbooks (per model).  
- Each workbook: first 62 columns = answers (Likert words: *strongly disagree, disagree, agree, strongly agree*).  
- Uses official conversion matrices to compute **Economic** (x) and **Social** (y) axes with offsets.  
- Outputs:
  - `combined_compass_scores.csv` (per‑response E/S)
  - `summary_compass_stats.csv` (means/SDs by model×role)
  - `welch_compass_results.csv` / `.xlsx` (pairwise Welch’s t (potentially with multiplicity adjustment) + Hedges’ g & 95% CI)
  - `political_compass_averages.pdf` and `.png` (quadrant scatter of role means)
  - `likert_distribution.csv` (per model×role choice frequencies)

### Sequential analysis (order effects)

File: `sequential_analysis.py`

- Inputs: one workbook produced by the sequential collector (or the same schema).  
- Analyzes Big Five Personality and Political Compass scores when **administered sequentially** (e.g., Big Five → Compass vs Compass → Big Five).  
- Outputs OCEAN/Compass summaries and the same Welch/Hedges’ g tables by order.


## Personas / roles

The collection and analysis assume the following role names and sheet labels:

- Code keys: `default`, `left-authoritarian`, `right-authoritarian`, `left-libertarian`, `right-libertarian`  
- Sheet names: `Default`, `Left-Authoritarian`, `Right-Authoritarian`, `Left-Libertarian`, `Right-Libertarian`

If your files use different names, either rename sheets or adapt the mapping in the scripts.

---

## Reproducibility notes

- We did not modify the model parameters (e.g., temperature) and kept the provider’s default setting, in order to reflect the “out-of-the-box” behavior as experienced by end-users.
- API behavior can change over time; record the **model name**, **date**, and **script commit**.
- For LLMs accessed **without an API** (e.g., local or open-source models), also record the **model version**, **weights checkpoint**, and **execution environment** (hardware, libraries, and OS) to ensure reproducibility.  

---

## License & citation

- Code: MIT License (feel free to reuse with attribution).  
- If you use this toolkit in academic work, please cite the repository and the Python ecosystem you rely on (e.g., “Python 3.10, NumPy, pandas, SciPy, statsmodels, Matplotlib”).

---

## FAQ

**Q: Do I have to use the included example data (.xlsx)?**  
A: No. You can point the analysis scripts to your own files or to remote URLs (e.g., Zenodo).

**Q: Can I add new models?**  
A: Yes. The framework is model-agnostic, but depending on the provider you may need to adjust the collection scripts:  

- **API-based models** (e.g., OpenAI, Anthropic, Mistral Cloud, DeepSeek):  
  - Update the script to call the correct client library and version (e.g., `openai==0.28`).  
  - Set the appropriate API key as an environment variable (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).  
  - Model responses are retrieved through the provider’s API endpoints and stored in CSV/XLSX.  

- **Open-source / local models** (e.g., Llama, Mistral, Falcon):  
  - Use inference frameworks such as **Hugging Face Transformers**, **vLLM**, or **Ollama** to query the model.  
  - Adapt the `get_response()` function in the scripts to call the local inference API instead of the OpenAI client.
    
Once the collector is adapted to your model, save the outputs and include the resulting CSV/XLSX files in the analysis step via the `--inputs` option.



