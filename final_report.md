# Final Report: Semantic Projection Analysis

## Methods

### Design and analytic objective
We quantified emotional language along a predefined semantic axis spanning **fear/anxiety** to **calm/optimism** and tested whether projection scores differed by experimental condition. Analyses were conducted at two levels: (1) **word-level** projections (all available emotion words), and (2) **participant-level mean** projections (mean across words per participant).

### Data sources and preprocessing
Two input files were used:
- `data/emotion_words_checked.csv`
- `data/Flow_current.csv`

Records were linked by participant `ID` after coercing `ID` to character in both files. Condition labels were merged from `Flow_current.csv` using a stripped condition column (`Exp_Condition`, with fallback handling for trailing whitespace in source headers).

For semantic projection, the emotion-word field was parsed as a list and cleaned by removing tokens prefixed with `not_` (e.g., `not_afraid`). Rows without usable word lists were excluded from projection steps.

### Embedding model and semantic axis construction
Embeddings were generated with a Sentence-Transformers pipeline wrapping:
- Transformer backbone: `siebert/sentiment-roberta-large-english`
- Pooling: mean token pooling

The fear-calm axis was built using anchor sets:
- Fear/anxiety pole: `fears`, `anger`, `afraid`, `fear`, `anxiety`, `pessimistic`, `panic`, `stress`, `preoccupied`, `fixated`
- Calm/optimism pole: `calm`, `quiet`, `serene`, `optimistic`, `upbeat`, `hopeful`, `confident`, `distracted`, `tune_out`

Context overrides were applied before encoding for two lexical items to improve contextualized representation:
- `tune_out` -> `tune out and relax`
- `distracted` -> `distracted from worries`

Let `a` be the mean embedding of fear/anxiety anchors and `b` the mean embedding of calm/optimism anchors. The semantic axis vector `u` was computed as the mean of all pairwise differences `(fear_i - calm_j)`. For each word embedding `v`, projection was computed as:

\[
\text{projection} = \frac{u \cdot v}{\|u\|}
\]

Higher values indicate stronger alignment with the fear/anxiety pole; lower values indicate stronger alignment with the calm/optimism pole.

### Outcome construction
- **Word-level outcome:** one projection score per word (`projection_fear_vs_calm`)
- **Participant-level outcome:** participant mean projection (`mean_projection_fear_vs_calm`)

Condition coding followed the analysis scripts, with condition `3` used as reference (Control), and contrasts estimated for condition `1` (Flow: VR + Meditation) and condition `2` (VR Only).

### Statistical models

#### Word-level model (primary mixed-effects specification)
A linear mixed-effects model was fit:

\[
\text{projection}_{ij} = \beta_0 + \beta_1\,\text{condition1}_i + \beta_2\,\text{condition2}_i + b_{0i} + \varepsilon_{ij}
\]

where `b0i` is a random intercept for participant.

Inference used **CR2 cluster-robust** variance estimation and Wald tests for pairwise contrasts:
- `1 vs ref` (Flow vs Control)
- `2 vs ref` (VR Only vs Control)
- `1 vs 2` (Flow vs VR Only)

A sensitivity analysis fit a participant-clustered linear model with CR2 correction; pairwise p-values were compared to mixed-model results.

#### Participant-mean model
A linear model was fit to participant means:

\[
\text{mean projection}_{i} = \beta_0 + \beta_1\,\text{condition1}_i + \beta_2\,\text{condition2}_i + \varepsilon_i
\]

Inference used **CR2 robust** standard errors. Pairwise contrasts were tested with CR2 Wald tests, and condition means were reported with t-based 95% confidence intervals.

### Software and reproducibility
Python environment targets documented in the project README:
- Python 3.12.12
- sentence-transformers 5.2.2
- transformers 5.0.0
- torch 2.9.0
- scikit-learn 1.6.1
- numpy 2.0.2
- pandas 2.2.2
- matplotlib 3.10.0

R analyses used `lme4`, `lmerTest`, `clubSandwich`, `emmeans`, `performance`, and `ggplot2`.

Output artifacts were generated in `semantic_projection/`, including:
- `semantic_projection_roberta.csv`
- `semantic_projection_roberta_mean.csv`
- coefficient and pairwise-comparison tables
- diagnostic text reports
- final figures (`semantic_projection_final.pdf`, `semantic_projection_final_mean.pdf`)

## Results

### Sample characteristics
- **Word-level dataset:** 117 projected words from 28 participants
  - Condition 1 (Flow): 33 words, 9 participants
  - Condition 2 (VR Only): 54 words, 12 participants
  - Condition 3 (Control): 30 words, 7 participants
- **Participant-mean dataset:** 28 participants
  - Condition 1: n = 9
  - Condition 2: n = 12
  - Condition 3: n = 7

### Condition means (participant-level projections)
Mean projection scores (lower = more calm/optimism-aligned):
- Control (3): mean = -19.473, 95% CI [-30.102, -8.844]
- Flow (1): mean = -28.697, 95% CI [-28.913, -28.482]
- VR Only (2): mean = -22.674, 95% CI [-29.021, -16.328]

### Word-level mixed-effects analysis (CR2 inference)
CR2-robust fixed effects (`projection_fear_vs_calm ~ condition + (1|Participant)`, ref = Control):
- Intercept (Control): \(\beta=-15.746\), SE = 4.803, p = 0.0247
- Flow vs Control: \(\beta=-12.893\), SE = 4.803, p = 0.0218
- VR Only vs Control: \(\beta=-6.647\), SE = 5.598, p = 0.2635

Pairwise CR2 Wald tests:
- Flow vs Control: p = 0.00727 (**)
- VR Only vs Control: p = 0.23509 (ns)
- Flow vs VR Only: p = 0.02999 (*)

Sensitivity check (LM + CR2) yielded highly similar pairwise inference:
- Flow vs Control: p = 0.00440
- VR Only vs Control: p = 0.18349
- Flow vs VR Only: p = 0.03053

### Participant-mean linear model (CR2 inference)
CR2-robust coefficients (`mean_projection_fear_vs_calm ~ condition`, ref = Control):
- Intercept (Control): \(\beta=-19.473\), SE = 4.344, p = 0.00418
- Flow vs Control: \(\beta=-9.225\), SE = 4.345, p = 0.05344
- VR Only vs Control: \(\beta=-3.202\), SE = 5.214, p = 0.55003

Pairwise CR2 Wald tests (participant means):
- Flow vs Control: p = 0.03374 (*)
- VR Only vs Control: not significant
- Flow vs VR Only: p = 0.03681 (*)

### Diagnostics summary
For the word-level mixed model:
- No singular fit and convergence acceptable.
- Heteroscedasticity and residual non-normality were flagged by diagnostics.
- Outlier check did not indicate influential outliers under the specified threshold.
- Variance explained was modest (marginal \(R^2\) = 0.067; conditional \(R^2\) = 0.108).

### Main empirical pattern
Across both analysis levels, **Condition 1 (Flow: VR + Meditation)** showed significantly more negative fear-calm projections than both Control and VR Only in pairwise CR2 tests, indicating stronger alignment with the calm/optimism end of the semantic axis. **Condition 2 (VR Only)** did not significantly differ from Control.
