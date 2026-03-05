# Flow Scripts

End-to-end analysis workspace for:
- semantic projection (RoBERTa-based embedding axis + word/participant modeling)
- VADER sentiment modeling (label-level and compound-level)

## Directory Structure

```text
flow_scripts/
  data/
    emotion_words_checked.csv
    Flow_current.csv

  semantic_projection/
    semantic_projection.py
    word_level.R
    mean_level.R
    requirements.txt
    ... generated CSV/PDF/report outputs

  sentiment/
    sentiment.py
    sentiment.R
    ... generated CSV/PDF/report outputs

  .venv/   (Python 3.12 local environment)
```

## Environment

## Python

Use the project virtual environment:

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
```

Pinned core packages (matching current setup):
- sentence-transformers==5.2.2
- transformers==5.0.0
- torch==2.9.0
- scikit-learn==1.6.1
- numpy==2.0.2
- pandas==2.2.2
- matplotlib==3.10.0
- vaderSentiment==3.3.2

Install/reinstall from semantic projection requirements + VADER:

```bash
python -m pip install -r semantic_projection/requirements.txt
python -m pip install vaderSentiment
```

## R

Required R packages:
- lme4
- lmerTest
- clubSandwich
- emmeans
- performance
- ggplot2
- dplyr
- tidyr
- jsonlite

Install once in R:

```r
install.packages(c(
  "lme4", "lmerTest", "clubSandwich", "emmeans", "performance",
  "ggplot2", "dplyr", "tidyr", "jsonlite"
))
```

## Run Order

Run from project root:

```bash
cd "$(git rev-parse --show-toplevel)"
```

1) Build semantic projection datasets and embedding plot

```bash
source .venv/bin/activate
python semantic_projection/semantic_projection.py
```

2) Run word-level semantic projection model

```bash
Rscript semantic_projection/word_level.R
```

3) Run mean-level semantic projection model

```bash
Rscript semantic_projection/mean_level.R
```

4) Build VADER sentiment CSV

```bash
source .venv/bin/activate
python sentiment/sentiment.py
```

5) Run sentiment models + plots (CR2-corrected)

```bash
Rscript sentiment/sentiment.R
```

## Path Behavior

- `semantic_projection/semantic_projection.py`
  - reads inputs from `data/`
  - writes outputs to `semantic_projection/semantic_projection/` (nested output directory by current script config)

- `semantic_projection/word_level.R` and `semantic_projection/mean_level.R`
  - read and write in `semantic_projection/`

- `sentiment/sentiment.py` and `sentiment/sentiment.R`
  - read/write in `sentiment/`

## Key Outputs

## Semantic projection
- `semantic_projection/semantic_projection_roberta.csv`
- `semantic_projection/semantic_projection_roberta_mean.csv`
- `semantic_projection/semantic_projection_final.pdf`
- `semantic_projection/semantic_projection_final_mean.pdf`
- `semantic_projection/analysis_report.txt`
- `semantic_projection/analysis_report_mean.txt`
- `semantic_projection/diagnostics_report.txt`
- `semantic_projection/diagnostics_report_mean.txt`

## Sentiment
- `sentiment/vader_sentiment_emotional_only.csv`
- `sentiment/sentiment_condition_label_plot.pdf`
- `sentiment/sentiment_compound_condition_plot.pdf`
- `sentiment/diagnostics_report_sentiment.txt`
- `sentiment/coefficients_sentiment_cr2.csv`
- `sentiment/coefficients_compound_cr2.csv`
- `sentiment/pairwise_sentiment_cr2.csv`
- `sentiment/pairwise_compound_cr2.csv`

## Notes

- CR2 robust corrections are applied in R where heteroscedasticity/non-normality was detected.
- If plots differ across machines, verify exact Python package versions and interpreter path.
- In VS Code, use interpreter:
  `.venv/bin/python`
