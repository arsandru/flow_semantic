# Flow Scripts Reproducibility Guide

This repository contains scripts for semantic projection analysis using:
- `siebert/sentiment-roberta-large-english`
- `sentence-transformers` mean pooling wrapper

## 1) Exact environment (match for reproducibility)

Use Python 3.12 and these exact package versions:

- Python: `3.12.12`
- sentence-transformers: `5.2.2`
- transformers: `5.0.0`
- torch: `2.9.0`
- scikit-learn: `1.6.1`
- numpy: `2.0.2`
- pandas: `2.2.2`
- matplotlib: `3.10.0`

## 2) Create environment

```bash
cd /Users/razvan/Documents/flow_scripts
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  sentence-transformers==5.2.2 \
  transformers==5.0.0 \
  torch==2.9.0 \
  scikit-learn==1.6.1 \
  numpy==2.0.2 \
  pandas==2.2.2 \
  matplotlib==3.10.0
```

## 3) Verify versions

```bash
python -V
python -m pip show sentence-transformers transformers torch scikit-learn numpy pandas matplotlib
```

## 4) Run analysis

Place these files in the working directory:
- `emotion_words_checked.csv`
- `Flow_current.csv`

Then run:

```bash
cd /Users/razvan/Documents/flow_scripts
source .venv/bin/activate
python semantic_projection.py
```

## 5) Expected outputs

Generated in the working directory:
- `semantic_projection_roberta.csv`
- `semantic_projection_roberta_mean.csv`
- `semantic_projection_roberta.pdf`

## 6) VS Code interpreter

In VS Code, select this interpreter for reproducibility:

`/Users/razvan/Documents/flow_scripts/.venv/bin/python`

## 7) Important reproducibility notes

- Keep the script identical across environments.
- Ensure only one active `encode_word` logic per execution path.
- Use the same input CSVs with identical contents.
- Differences in Python/package versions can change embeddings and PCA outputs.

