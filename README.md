# FAMICOM Complexity & Familiarity Analysis

A comprehensive framework for analyzing prompt complexity and familiarity metrics using the **FAMICOM** methodology. This project replicates the FAMICOM paper's findings across multiple language models (Phi-3-mini, Mistral-7B) and datasets, enabling correlation analysis between prompt characteristics and model performance.

## Table of Contents

- [Project Overview](#project-overview)
- [FAMICOM Methodology](#famicom-methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Pipeline](#basic-pipeline)
  - [Phi-3 Replication](#phi-3-replication)
  - [AALC Replication](#aalc-replication)
  - [Analysis & Correlation](#analysis--correlation)
- [Key Components](#key-components)
- [Results & Outputs](#results--outputs)
- [Requirements](#requirements)
- [Configuration](#configuration)

## Project Overview

This project implements a complete replication of the **FAMICOM** framework—a metric that combines **FAMiliarity** and **COMplexity** scores to predict language model performance on diverse question-answering tasks. The framework enables:

✅ **Prompt Complexity Analysis** – Quantifies problem difficulty using guided reasoning steps  
✅ **Familiarity Measurement** – Combines perplexity and keyword similarity to assess prompt familiarity  
✅ **FAMICOM Score Computation** – Integrates both metrics via the formula: `FAMICOM = f^a × c^(-b)`  
✅ **Performance Correlation** – Validates correlation between FAMICOM scores and model accuracy  
✅ **Multi-Model Support** – Supports Phi-3-mini, Mistral-7B, and other models  
✅ **Multi-Dataset Support** – Includes MMLU, StrategyQA, BigBench, CommonSenseQA, and more  

## FAMICOM Methodology

### Core Formula

```
FAMICOM = familiarity^a × complexity^(-b)
```

Where:
- **f (familiarity)** – How well a language model is already familiar with the prompt topic (0 to 1)
- **c (complexity)** – The intrinsic difficulty of reasoning required (typically 1 to 5+)
- **a** – Exponent for familiarity (default: 1.0)
- **b** – Exponent for complexity (default: 1.0)

### Interpretation
- **Higher FAMICOM** → Easier question (high familiarity, low complexity) → Higher expected accuracy
- **Lower FAMICOM** → Harder question (low familiarity, high complexity) → Lower expected accuracy

### Familiarity Score (f)
Computed as the maximum of two metrics:
1. **Token Similarity via Perplexity** – Text perplexity relative to a base model
2. **Keyword Similarity** – Semantic coherence of extracted keywords

### Complexity Score (c)
Estimated via **guided complexity** – the model counts the number of reasoning steps required:
- Simple factual recall = 2 steps
- Basic arithmetic = 3 steps
- Multi-step reasoning = 4+ steps

## Project Structure

```
cancer-chatbot/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── combined_FAMCOM.py                 # Main FAMICOM score computation
├── correlation_analysis.py            # Spearman correlation & visualization
│
├── basic_pipeline/                    # Core analysis pipeline
│   ├── generate_prompts.py           # Prompt preparation
│   ├── calculate_famicom.py          # FAMICOM calculation (with model results)
│   ├── analyze_famicom_correlation.py # Correlation analysis
│   ├── analyze_complexity_correlation.py
│   ├── analyze_correlations.py
│   ├── complexity_analyzer_*.py      # Complexity measurement variants
│   └── *.json                         # Cached results & scores
│
├── famicom-replication-phi/           # Phi-3-mini model replication
│   ├── accuracy_script.py             # Compute model accuracy
│   ├── generate_prompts.py            # Phi-specific prompt generation
│   ├── phi3_model_answers.json        # Model predictions
│   ├── prompts_with_accuracy.json     # Prompts + accuracy labels
│   │
│   ├── complexity/
│   │   ├── complexity_analyzer_phi.py # Phi-3 guided complexity measurement
│   │   └── complexity_results_phi.json# Cached complexity scores
│   │
│   ├── familiarity/
│   │   ├── perplexity.py              # Perplexity calculation (HF + vLLM)
│   │   ├── similarity.py              # Keyword similarity measurement
│   │   ├── runner.py                  # Familiarity pipeline orchestration
│   │   └── __pycache__/
│   │
│   ├── scripts/
│   │   └── eval_models.py             # Model evaluation utilities
│   │
│   ├── data/
│   │   ├── prepared_prompts.json      # Standardized input prompts
│   │   └── cot_hub/                   # Chain-of-Thought datasets
│   │       ├── mmlu.json
│   │       ├── strategyqa.json
│   │       ├── bigbench.json
│   │       ├── commonsenseqa.json
│   │       └── ...
│   │
│   └── outputs/
│       └── famicom_combined_phi.json  # Final FAMICOM results
│
├── famicom-replication-aalc/          # AALC dataset replication
│   ├── generate_prompts.py
│   ├── data/
│   │   ├── prepared_prompts.json
│   │   └── cot_hub/                   # Chain-of-Thought datasets
│   │       ├── anli.json
│   │       ├── arc_challenge.json
│   │       ├── bigbench.json
│   │       ├── copa.json
│   │       ├── hotpotqa_mc.json
│   │       ├── math_gsm8k.json
│   │       ├── mmlu.json
│   │       ├── openbookqa.json
│   │       ├── piqa.json
│   │       └── socialiqa.json
│   │
│   └── __pycache__/
│
├── data/
│   └── prepared_prompts.json          # Master prompt repository
│
└── outputs/
    ├── famicom_combined_phi.json      # Phi-3 FAMICOM scores
    └── familiarity/
        └── phi3_transformers_64.json  # Phi-3 familiarity scores
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration)
- Sufficient disk space (~20GB for models)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sameerashahh/cancer-chatbot.git
   cd cancer-chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLP models:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **(Optional) For Phi-3 replication – Download model:**
   ```bash
   # Phi-3-mini will auto-download on first use via Hugging Face
   huggingface-cli login  # Authenticate with HF token
   ```

## Usage

### Basic Pipeline

The basic pipeline demonstrates the core FAMICOM workflow:

#### 1. Calculate FAMICOM Scores
```bash
cd basic_pipeline
python calculate_famicom.py
```
**Output:** `famicom_scores.json`

#### 2. Analyze Correlation with Model Performance
```bash
python analyze_famicom_correlation.py
```
**Output:** Spearman correlation coefficient + visualization

#### 3. Complexity-Only Analysis
```bash
python complexity_analyzer_fixed.py
python analyze_complexity_correlation.py
```

### Phi-3 Replication

Complete FAMICOM replication using **Phi-3-mini-128k-instruct** model.

#### Step 1: Prepare Prompts
```bash
cd famicom-replication-phi
python generate_prompts.py
```
Creates `data/prepared_prompts.json` from CoT-Hub datasets.

#### Step 2: Measure Complexity (Guided)
```bash
cd complexity
python complexity_analyzer_phi.py
```
**Output:** `complexity_results_phi.json`

#### Step 3: Measure Familiarity

**a) Extract Keywords:**
```bash
cd ../familiarity
python similarity.py
```

**b) Calculate Perplexity:**
```bash
python perplexity.py
```

**c) Run Full Familiarity Pipeline:**
```bash
python runner.py
```
**Output:** `../outputs/familiarity/phi3_transformers_64.json`

#### Step 4: Compute Model Accuracy
```bash
cd ../
python accuracy_script.py
```
**Output:** `prompts_with_accuracy.json`

#### Step 5: Combine into FAMICOM Scores
```bash
cd ../../
python combined_FAMCOM.py
```
**Output:** `outputs/famicom_combined_phi.json`

#### Step 6: Analyze Correlation
```bash
python correlation_analysis.py
```
**Output:** Spearman correlation + trend plot

### AALC Replication

Alternative replication using the AALC dataset collection:

```bash
cd famicom-replication-aalc
python generate_prompts.py
# Then follow similar steps as Phi-3 replication
```

### Analysis & Correlation

Run the main correlation analysis:
```bash
python correlation_analysis.py
```

This generates:
- **Spearman correlation coefficient (ρ)** between FAMICOM and accuracy
- **P-value** for statistical significance
- **Trend visualization** binned by FAMICOM score ranges

## Key Components

### 1. Complexity Measurement (`complexity_analyzer_phi.py`)

Measures guided complexity using the model's own reasoning:

```python
# Example: Count reasoning steps
prompt = "Solve: If John has 5 apples and Mary has 3, how many do they have together?"
steps = [
    "1) Parse the numbers from the question",
    "2) Identify the operation (addition)",
    "3) Compute the sum (5 + 3 = 8)",
    "4) Format the answer"
]
complexity_score = len(steps)  # 4
```

**Guided Complexity Levels:**
- **2 steps:** Simple factual recall (capitals, definitions)
- **3 steps:** Arithmetic, basic logic
- **4 steps:** Multi-step reasoning, synthesis
- **5+ steps:** Complex chains, cross-domain reasoning

### 2. Familiarity Measurement

#### Perplexity (HF Transformers)
```python
from familiarity.perplexity import compute_perplexity_transformers

results = compute_perplexity_transformers(
    texts=["What is Python?"],
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
)
# Returns: PerplexityResult with ppl, tokens, logprobs
```

#### Keyword Similarity (spaCy + Sentence-Transformers)
```python
from familiarity.similarity import compute_keyword_similarity

results = compute_keyword_similarity(
    texts=["Machine learning algorithms process data efficiently"],
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
# Returns: SimilarityResult with keywords and mean pairwise similarity
```

### 3. FAMICOM Combination (`combined_FAMCOM.py`)

Integrates all metrics:

```python
# Parameters
A = 1.0  # exponent for familiarity
B = 1.0  # exponent for complexity

# Combine
famicom_score = (familiarity_score ** A) * (complexity_score ** (-B))
```

### 4. Correlation Analysis

Validates FAMICOM's predictive power:

```python
# Compute Spearman correlation
corr, p_value = spearmanr(famicom_scores, model_accuracies)
# Expected: positive correlation (p < 0.05)
```

## Results & Outputs

### Expected Outputs

After running the full pipeline, you'll have:

1. **FAMICOM Scores** (`outputs/famicom_combined_phi.json`)
   ```json
   [
     {
       "prompt": "What is the capital of France?",
       "familiarity": 0.8234,
       "complexity": 2.0,
       "famicom_score": 0.4117
     },
     ...
   ]
   ```

2. **Correlation Results**
   - Spearman ρ (typically 0.3–0.6 for good models)
   - P-value (p < 0.05 for significance)
   - Trend plot showing accuracy vs. FAMICOM bins

3. **Accuracy Data** (`prompts_with_accuracy.json`)
   - Model predictions
   - Ground truth labels
   - Per-prompt accuracy

### Key Findings

The FAMICOM metric successfully predicts model performance:
- **Positive Spearman correlation** between FAMICOM and accuracy
- **Statistical significance** (p < 0.05)
- **Clear trend** in binned visualization

Example result:
```
Spearman correlation: 0.45 (p = 0.002) **
```

## Requirements

**Core Dependencies:**
```
transformers>=4.42.0      # HuggingFace model loading
torch>=2.1.0              # Deep learning framework
sentence-transformers>=3.0.1  # Semantic embeddings
spacy>=3.7.4              # NLP tokenization & keyword extraction
scikit-learn>=1.4.2       # Statistical analysis
vllm>=0.5.0               # Fast LLM inference
numpy>=1.26.4             # Numerical computing
pandas>=2.2.2             # Data manipulation
ujson>=5.10.0             # Fast JSON parsing
```

Install all via:
```bash
pip install -r requirements.txt
```

## Configuration

### FAMICOM Parameters

Edit exponents in `combined_FAMCOM.py` or individual scripts:

```python
A = 1.0  # Familiarity exponent
B = 1.0  # Complexity exponent (negative in formula)
```

**Recommended ranges:**
- `A ∈ [0.5, 2.0]` – Adjust familiarity sensitivity
- `B ∈ [0.5, 2.0]` – Adjust complexity sensitivity

### Model Selection

Change model in complexity/familiarity scripts:

```python
# Complexity
MODEL_NAME = "microsoft/phi-3-mini-128k-instruct"  # or Mistral-7B-Instruct-v0.2

# Familiarity
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # HF Transformers
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Sentence embeddings
```

### Dataset Selection

Place `.json` files in `data/cot_hub/` directory. Supported datasets:
- MMLU, StrategyQA, BigBench
- CommonSenseQA, COPA, PIQA
- ANLI, ARC Challenge, HotpotQA
- OpenBookQA, SocialIQA, GSM8K

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size in complexity/familiarity scripts
batch_size = 1  # Default: 4
max_tokens = 10  # Reduce generation length
NUM_PROMPTS = 500  # Start with fewer prompts
```

### CUDA Not Available
```bash
# Force CPU mode (slow!)
device = "cpu"  # In perplexity.py and scripts
```

### Missing spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### Model Download Issues
```bash
# Set HF cache directory
export HF_HOME=/path/to/cache
huggingface-cli login  # Provide token
```

## Citation

If you use this framework, please cite:

```bibtex
@article{famicom2024,
  title={FAMICOM: Towards Predicting Language Model Performance via Complexity and Familiarity},
  year={2024}
}
```

## License

This project is open source and available under the MIT License.

## Contact & Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Last Updated:** November 11, 2025

