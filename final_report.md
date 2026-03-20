# Final Report: Semantic Projection Analysis

## Methods

### Design and analytic objective
We quantified emotional language along a predefined semantic axis spanning **distress** to **relaxation** and tested whether projection scores differed by experimental condition. Analyses were conducted at two levels: (1) **word-level** projections (all available emotion words), and (2) **participant-level mean** projections (mean across words per participant).

### Data sources and preprocessing
Two input files were used:
- `data/emotion_words_checked.csv`
- `data/Flow_current.csv`

Records were linked by participant `ID` after coercing `ID` to character in both files. Condition labels were merged from `Flow_current.csv` using a stripped condition column (`Exp_Condition`, with fallback handling for trailing whitespace in source headers).

For semantic projection, the emotion-word field was parsed as a list and cleaned by removing tokens prefixed with `not_` (e.g., `not_afraid`). Before encoding, underscore-delimited tokens were converted to space-delimited text so that multiword expressions were represented in natural language form. Rows without usable word lists were excluded from projection steps.

### Embedding model and semantic axis construction
Embeddings were generated with a Sentence-Transformers pipeline wrapping:
- Transformer backbone: `siebert/sentiment-roberta-large-english`
- Pooling: mean token pooling

The distress-relaxed axis was built using anchor sets:
- Distress pole: `anxiety`, `anxious`, `nervous`, `worried`, `bothered`, `uncomfortable`, `distressed`, `tense`, `uneasy`, `overwhelmed`
- Relaxed pole: `calm`, `relaxed`, `relaxing`, `serene`, `peaceful`, `comfortable`, `comforting`, `at ease`, `soothed`, `settled`

Context overrides were applied before encoding for two lexical items to improve contextualized representation:
- `tune_out` -> `tune out and relax`
- `distracted` -> `distracted from worries`

Let `a` be the mean embedding of distress anchors and `b` the mean embedding of relaxed anchors. The semantic axis vector `u` was computed as the mean of all pairwise differences `(distress_i - relaxed_j)`. For each word embedding `v`, projection was computed as:

\[
\text{projection} = \frac{u \cdot v}{\|u\|}
\]

Higher values indicate stronger alignment with the distress pole; lower values indicate stronger alignment with the relaxed pole.

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
Mean projection scores (lower = more relaxed-aligned):
- Control (3): mean = -19.749, 95% CI [-30.228, -9.270]
- Flow (1): mean = -29.056, 95% CI [-29.188, -28.924]
- VR Only (2): mean = -22.882, 95% CI [-29.196, -16.568]

### Word-level mixed-effects analysis (CR2 inference)
CR2-robust fixed effects (`projection_fear_vs_calm ~ condition + (1|Participant)`, ref = Control):
- Intercept (Control): \(\beta=-15.961\), SE = 4.732, p = 0.0228
- Flow vs Control: \(\beta=-13.065\), SE = 4.733, p = 0.0191
- VR Only vs Control: \(\beta=-6.632\), SE = 5.536, p = 0.2597

Pairwise CR2 Wald tests:
- Flow vs Control: p = 0.00577, Bonferroni-adjusted p = 0.01731
- VR Only vs Control: p = 0.23091, Bonferroni-adjusted p = 0.69273
- Flow vs VR Only: p = 0.02515, Bonferroni-adjusted p = 0.07545

Sensitivity check (LM + CR2) yielded highly similar pairwise inference:
- Flow vs Control: p = 0.00362
- VR Only vs Control: p = 0.18495
- Flow vs VR Only: p = 0.02576

### Participant-mean linear model (CR2 inference)
CR2-robust coefficients (`mean_projection_fear_vs_calm ~ condition`, ref = Control):
- Intercept (Control): \(\beta=-19.749\), SE = 4.283, p = 0.00365
- Flow vs Control: \(\beta=-9.307\), SE = 4.283, p = 0.04878
- VR Only vs Control: \(\beta=-3.133\), SE = 5.154, p = 0.55403

Pairwise CR2 Wald tests (participant means):
- Flow vs Control: p = 0.02977, Bonferroni-adjusted p = 0.08931
- VR Only vs Control: p = 0.54330, Bonferroni-adjusted p = 1.00000
- Flow vs VR Only: p = 0.03140, Bonferroni-adjusted p = 0.09419

### Diagnostics summary
For the word-level mixed model:
- No singular fit and convergence acceptable.
- Heteroscedasticity and residual non-normality were flagged by diagnostics.
- Outlier check did not indicate influential outliers under the specified threshold.
- Variance explained was modest (marginal \(R^2\) = 0.069; conditional \(R^2\) = 0.104).

### Main empirical pattern
At the word level, **Condition 1 (Flow: VR + Meditation)** showed more negative distress-relaxed projections than Control in both the raw and Bonferroni-corrected pairwise CR2 tests. The raw Flow versus VR Only contrast was also significant, but it did not remain significant after Bonferroni correction. At the participant-mean level, the same directional pattern was observed, with the Flow coefficient reaching nominal significance and the pairwise contrasts remaining marginal after Bonferroni correction. **Condition 2 (VR Only)** did not significantly differ from Control at either level.

### Embedding-model robustness
As exploratory robustness checks, the word-level semantic projection pipeline was repeated using two alternative embedding spaces: `Qwen/Qwen3-Embedding-0.6B`, representing a different embedding-model family, and `j-hartmann/emotion-english-roberta-large`, representing an alternative affective fine-tuning of the RoBERTa architecture. Across all three models, the primary Flow versus Control contrast remained significant after Bonferroni correction (RoBERTa: adjusted \(p = 0.0173\); Qwen: adjusted \(p = 0.00184\); emotion-tuned RoBERTa: adjusted \(p = 0.0178\)). The secondary contrasts were less stable across models and did not survive Bonferroni correction. Thus, the principal finding that the Flow condition was more strongly aligned with the relaxed pole than Control was robust both to embedding architecture and to the specific affective fine-tuning used, whereas finer-grained contrasts between the intervention conditions were model-sensitive.
