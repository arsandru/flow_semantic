import numpy as np
import pandas as pd
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load emotionally sensitive embedding model
# -----------------------------
# This wraps siebert/sentiment-roberta-large-english into a SentenceTransformer-style encoder
from sentence_transformers import SentenceTransformer, models

MODEL_ID = "siebert/sentiment-roberta-large-english"

# Build explicit ST pipeline (instead of auto-conversion)
word_embedding_model = models.Transformer(MODEL_ID)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "semantic_projection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_word(w: str) -> np.ndarray:
    """Encode a single word into a vector, with context overrides."""
    if not isinstance(w, str):
        return model.encode(str(w))

    key = w.strip().lower()
    if key == "tune_out":
        w = "tune out and relax"
    elif key == "distracted":
        w = "distracted from worries"

    return model.encode(w)


# -----------------------------
# 2. Feature subspace builder (same logic as getSemanticSubspace)
# -----------------------------
def get_semantic_subspace(pos_words, neg_words, encode_fn):
    """
    Implements the MATLAB getSemanticSubspace logic:

        - aVec = mean of positive-end vectors
        - bVec = mean of negative-end vectors
        - diffVec = mean of all pairwise (pos - neg) differences

    Parameters
    ----------
    pos_words : list of str
        Words representing one extreme of the feature (e.g. fear-side).
    neg_words : list of str
        Words representing the opposite extreme (e.g. calm-side).
    encode_fn : callable
        Function mapping a string to its embedding vector.

    Returns
    -------
    a_vec : np.array
        Mean vector of positive words.
    b_vec : np.array
        Mean vector of negative words.
    diff_vec : np.array
        Mean of all pairwise differences (pos - neg): the 1D feature axis.
    """
    pos_vecs = [encode_fn(w) for w in pos_words]
    neg_vecs = [encode_fn(w) for w in neg_words]

    pos_vecs = np.vstack(pos_vecs)
    neg_vecs = np.vstack(neg_vecs)

    a_vec = pos_vecs.mean(axis=0)
    b_vec = neg_vecs.mean(axis=0)

    diff_mat = np.array([p - n for p in pos_vecs for n in neg_vecs])
    diff_vec = diff_mat.mean(axis=0)

    return a_vec, b_vec, diff_vec

# ---------- Example: fear/anxiety vs calm/optimistic axis ----------
fear_anx_words = [
    'fears', 'anger', 'afraid', "fear", "anxiety", 'pessimistic', 'panic', 'stress', 'preoccupied', 'fixated'
]
calm_opt_words = [
    "calm", 'quiet', 'serene', "optimistic", "upbeat", 'hopeful', 'confident', 'distracted', 'tune_out'
]

a_vec, b_vec, fear_calm_axis = get_semantic_subspace(
    fear_anx_words,
    calm_opt_words,
    encode_word
)

axis_vec = fear_calm_axis
axis_norm = np.linalg.norm(axis_vec)

# ---------- Projection function (same as paper) ----------
def semantic_projection(word, axis_vec, axis_norm, encode_fn):
    """
    Project a single word onto the given semantic axis:
        projection = (axis · v_word) / ||axis||
    """
    v = encode_fn(word)
    if axis_norm == 0:
        return np.nan
    return float(axis_vec @ v / axis_norm)

# -----------------------------
# 3. Trial-level projection: per-word + participant mean
# -----------------------------
def process_trial_single_axis(word_list, axis_vec, axis_norm, encode_fn):
    """
    word_list: list of emotion words (strings) for one participant.
    axis_vec:  1D numpy array = fear_vs_calm axis (diffVec).
    axis_norm: scalar = ||axis_vec||.
    encode_fn: function(word) -> vector.

    Returns:
        {
          'word_projections': list of dicts { 'word': w, 'projection': proj },
          'mean_projection': scalar mean over all available words
        }
    """
    word_projs = []

    for w in word_list:
        if not isinstance(w, str):
            continue

        proj = semantic_projection(w, axis_vec, axis_norm, encode_fn)
        word_projs.append({"word": w, "projection": proj})

    if not word_projs:
        return {
            "word_projections": [],
            "mean_projection": np.nan
        }

    mean_proj = float(np.mean([wp["projection"] for wp in word_projs]))

    return {
        "word_projections": word_projs,
        "mean_projection": mean_proj
    }

# -----------------------------
# 4. Load CSVs and clean
# -----------------------------
df = pd.read_csv(DATA_DIR / "emotion_words_checked.csv")
df_t = pd.read_csv(DATA_DIR / "Flow_current.csv")

# Ensure ID types match
df["ID"] = df["ID"].astype(str)
df_t["ID"] = df_t["ID"].astype(str)

# Clean column names in df_t (strip whitespace / NBSP)
df_t.columns = df_t.columns.str.strip()

# Guess the condition column name
condition_col = "Exp_Condition" if "Exp_Condition" in df_t.columns else "Exp_Condition "

# Merge to reliably attach condition to each row in df
merged = df.merge(df_t[["ID", condition_col]], on="ID", how="left")

# Remove "not_" words
def remove_not_words(cell):
    if isinstance(cell, list):
        return [w for w in cell if isinstance(w, str) and not w.startswith("not_")]
    if isinstance(cell, str):
        try:
            words = ast.literal_eval(cell)
            if isinstance(words, list):
                return [w for w in words if isinstance(w, str) and not w.startswith("not_")]
            return []
        except Exception:
            return []
    return []


merged["text"] = merged["text"].apply(remove_not_words)

# -----------------------------
# 5. Loop over merged rows: compute projections
# -----------------------------
word_level_rows = []
participant_level_rows = []

for i, row in merged.iterrows():
    participant_id = row["ID"]
    condition = row[condition_col]
    word_list = row["text"]

    if not word_list:
        continue

    result = process_trial_single_axis(word_list, axis_vec, axis_norm, encode_word)

    # word-level projections
    for wp in result["word_projections"]:
        word_level_rows.append({
            "Participant": participant_id,
            "condition": condition,
            "word": wp["word"],
            "projection_fear_vs_calm": wp["projection"]
        })

    # participant-level mean projection
    participant_level_rows.append({
        "Participant": participant_id,
        "condition": condition,
        "mean_projection_fear_vs_calm": result["mean_projection"],
        "word_list": str(word_list)
    })


word_level_df = pd.DataFrame(word_level_rows)
participant_level_df = pd.DataFrame(participant_level_rows)
word_level_df = word_level_df.dropna(subset=["condition"])

participant_level_df = participant_level_df.dropna(subset=["condition"])
word_level_df.to_csv(OUTPUT_DIR / "semantic_projection_roberta.csv", index=False)
participant_level_df.to_csv(OUTPUT_DIR / "semantic_projection_roberta_mean.csv", index=False)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- 6. 2D semantic projection plot (axis + orthogonal PC1) ----
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Helper to encode a single word with RoBERTa
def encode_word(w: str) -> np.ndarray:
    return model.encode(w)

# ---- 0. Choose the DataFrame ----
data_for_plot = merged.copy()
data_for_plot = data_for_plot.dropna(subset=["Exp_Condition"])

# ---- 1. Collect all words, their embeddings (RoBERTa), and conditions ----
all_words = []
all_vecs = []
all_conditions = []

embedding_cache = {}  # to avoid re-encoding the same word many times

for idx, row in data_for_plot.iterrows():
    participant_id = row["ID"]
    condition = row["Exp_Condition"]
    word_list = row["text"]

    if not word_list:
        continue

    for w in word_list:
        if not isinstance(w, str):
            continue
        key = w.strip().lower()
        if w not in embedding_cache:
            
                embed_text = w
                if key in {"tune_out", "tune out"}:
                    embed_text = "tune out and relax"
                elif key == "distracted":
                    embed_text = "distracted from worries"

                embedding_cache[w] = encode_word(embed_text)
        all_words.append(w)
        all_vecs.append(embedding_cache[w])
        all_conditions.append(condition)

if not all_vecs:
    raise ValueError("No words with embeddings found for plotting.")

all_vecs = np.vstack(all_vecs)  # shape: (N_words, emb_dim)

# ---- 2. Build axis-aligned coordinates (using RoBERTa-based axis) ----
axis_vec = fear_calm_axis       # this should already be built using the same model
axis_norm = np.linalg.norm(axis_vec)
axis_unit = axis_vec / axis_norm

# X coordinate: semantic projection on fear–calm axis (same as paper)
x_coord = (all_vecs @ axis_vec) / axis_norm   # shape: (N_words,)

# Residual vectors: remove the axis-aligned component
proj_along_unit = all_vecs @ axis_unit        # scalar projection onto unit axis
residuals = all_vecs - np.outer(proj_along_unit, axis_unit)  # shape: (N_words, emb_dim)

# ---- 3. PCA on residuals to get 1D orthogonal semantic coordinate ----
pca = PCA(n_components=1)
y_coord = pca.fit_transform(residuals).ravel()   # shape: (N_words,)

# ---- 4. Plot: x = projection on fear–calm, y = orthogonal PC1, colored by condition ----
plt.figure(figsize=(10, 8))

all_conditions_arr = np.array(all_conditions)
unique_conditions = sorted(set(all_conditions_arr))
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_conditions)))

for cond, col in zip(unique_conditions, colors):
    mask = all_conditions_arr == cond
    plt.scatter(
        x_coord[mask],
        y_coord[mask],
        color=col,
        alpha=0.7,
        label=str(cond),
        edgecolor="none"
    )

# Optional: annotate words (can get busy if many)
for i, w in enumerate(all_words):
    plt.text(
        x_coord[i] + 0.01,
        y_coord[i] + 0.01,
        w,
        fontsize=7,
        alpha=0.6
    )

plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
plt.title("Semantic Projection of Emotional Words on Fear–Calm Axis (RoBERTa)")
plt.xlabel("Projection on fear_vs_calm axis  ( (u·v) / ||u|| )")
plt.ylabel("Orthogonal semantic variation (PC1 of residuals)")
plt.legend(title="Condition")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "semantic_projection_roberta.pdf")
plt.show()
