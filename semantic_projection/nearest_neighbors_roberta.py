import re
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "siebert/sentiment-roberta-large-english"
DEFAULT_TARGETS = ["distressed", "relaxed"]
TOP_K = 10
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    return tokenizer, model


def normalize_vocab_token(token: str) -> str | None:
    token = token.replace("Ġ", "").strip()
    if not token or not WORD_RE.fullmatch(token):
        return None
    return token.lower()


def build_candidate_matrix(tokenizer, model):
    embedding_weight = model.get_input_embeddings().weight.detach().cpu().numpy()
    vocab = tokenizer.get_vocab()

    seen = set()
    labels = []
    indices = []

    for token, idx in sorted(vocab.items(), key=lambda item: item[1]):
        label = normalize_vocab_token(token)
        if label is None or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        indices.append(idx)

    matrix = embedding_weight[np.array(indices)]
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return labels, matrix


def target_vector(tokenizer, model, text: str) -> np.ndarray:
    pieces = tokenizer.tokenize(text)
    if not pieces:
        raise ValueError(f"Could not tokenize target: {text}")

    piece_ids = tokenizer.convert_tokens_to_ids(pieces)
    emb = model.get_input_embeddings().weight[piece_ids].detach().cpu().numpy()
    vec = emb.mean(axis=0)
    return vec / np.linalg.norm(vec)


def nearest_neighbors(tokenizer, model, labels, matrix, target: str, top_k: int):
    vec = target_vector(tokenizer, model, target)
    sims = matrix @ vec
    order = np.argsort(-sims)

    neighbors = []
    target_norm = target.lower()
    for idx in order:
        label = labels[idx]
        if label == target_norm:
            continue
        neighbors.append((label, float(sims[idx])))
        if len(neighbors) == top_k:
            break
    return neighbors


def main() -> int:
    targets = sys.argv[1:] or DEFAULT_TARGETS

    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as exc:
        print("Failed to load the local RoBERTa model/tokenizer.", file=sys.stderr)
        print("Make sure `siebert/sentiment-roberta-large-english` is available in the local cache.", file=sys.stderr)
        print(f"Underlying error: {exc}", file=sys.stderr)
        return 1

    labels, matrix = build_candidate_matrix(tokenizer, model)

    for target in targets:
        print(f"\nNearest neighbors in RoBERTa vocabulary space for {target}:")
        for word, similarity in nearest_neighbors(tokenizer, model, labels, matrix, target, TOP_K):
            print(f"{word}\t{similarity:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
