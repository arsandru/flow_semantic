# Comprehensive Methodology & Code Review

## 1. Methodology Audit

### **A. Semantic Axis Construction & Embedding**
* **Model Selection**: The study utilizes `siebert/sentiment-roberta-large-english` to generate word-level and participant-level embeddings.
* **Axis Construction**: It constructs a 1D semantic axis spanning "fear/anxiety" to "calm/optimism" by calculating the mean of all pairwise differences between anchor word sets.
* **Contextual Refinement**: "Context overrides" are applied to polysemous terms (e.g., `tune_out` is encoded as `tune out and relax`) to ensure embeddings accurately reflect the intended psychological state.
* **Confidence Rating: 95% (High)**. The projection mathematics are standard for semantic subspace analysis, and the context cleaning is rigorous.

### **B. Statistical Modeling**
* **Approach**: A Linear Mixed-Effects (LME) model fits word-level projections with a random intercept for participants.
* **Robustness**: Analysis employs **CR2 Cluster-Robust Variance Estimation** via the `clubSandwich` package to account for participant-level clustering and heteroscedasticity.
* **Sensitivity Analysis**: Comparison between the Mixed Model and a participant-clustered Linear Model (LM) showed highly similar results.
* **Confidence Rating: 90% (High)**. CR2 is an industry-standard correction for small samples ($N=28$) with non-normal residuals.

### **C. Summary of Empirical Findings**
| Level | Contrast | Result | P-Value (CR2) |
| :--- | :--- | :--- | :--- |
| **Word-Level** | Flow vs. Control | Significant (**) | $p=0.007$ |
| **Word-Level** | Flow vs. VR Only | Significant (*) | $p=0.030$ |
| **Mean-Level** | Flow vs. Control | Significant (*) | $p=0.034$ |

---

## 2. Code Audit

### **A. Data Processing (`semantic_projection.py` & `sentiment.py`)**
* **Embedding Pipeline**: The code explicitly builds a SentenceTransformer pipeline with mean token pooling.
* **Negation Handling**: The `remove_not_words` function prevents negated tokens (e.g., `not_afraid`) from being incorrectly projected onto the axis.
* **VADER Integration**: For sentiment analysis, underscores are replaced with spaces so the engine can recognize natural language negations.
* **Confidence Rating: 95% (High)**. Logic is mathematically sound and follows modern NLP best practices.

### **B. Statistical Implementation (`word_level.R` & `mean_level.R`)**
* **Path Integrity**: R scripts dynamically detect the project directory, ensuring portability across local environments.
* **Automated Diagnostics**: Scripts capture singularity, convergence, heteroscedasticity, and normality, outputting them into detailed text reports.
* **Inference Logic**: Pairwise contrasts are tested using Wald tests with the CR2-corrected covariance matrix.
* **Confidence Rating: 90% (High)**. This is a high-quality implementation of robust standard errors for nested data.

### **C. Reproducibility & Environment**
* **Dependency Management**: The repository includes a `requirements.txt` pinning exact versions (e.g., `sentence-transformers==5.2.2`) to avoid drift in embedding values.
* **Documentation**: Detailed run-orders and environment setup instructions are provided in the README files.
* **Confidence Rating: 95% (High)**. The pipeline is well-documented and effectively "plug-and-play".

---

**Overall Audit Conclusion**: The analysis is technically sophisticated and statistically conservative. By prioritizing robust estimation (CR2) and semantic precision (context overrides), the repository provides high-confidence evidence that the Flow condition significantly shifts participant language toward the "Calm/Optimistic" pole of the semantic space.
