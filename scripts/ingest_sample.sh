#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ingest sample documents into the vector DB (FAISS + SQLite).
# - Reads from data/sample_docs.csv
# - Embeds text with SentenceTransformers
# - Stores vectors in FAISS index
# - Stores metadata in SQLite
# ---------------------------------------------------------------------------

set -euo pipefail  # strict mode: fail on errors/undefined vars/pipelines

python - <<'PY'
import pandas as pd
from pathlib import Path

# Project modules (vector index + embeddings + metadata utils)
from src.embed import embed_texts
from src.indexer import (
    load_index, create_index, add_vectors, save_index, load_doc_ids
)
from src.utils import ensure_metadata_table

# ---------------------------------------------------------------------------
# Step 1. Load sample data
# ---------------------------------------------------------------------------
data_path = Path("data/sample_docs.csv")
if not data_path.exists():
    raise FileNotFoundError(f"Missing input file: {data_path}")

df = pd.read_csv(data_path)

# Extract fields
texts = df['text'].astype(str).tolist()
ids = df['id'].astype(str).tolist()
metas = [
    {
        "title": t,
        "text": txt,
        "source": "sample",
        "created_at": ""
    }
    for t, txt in zip(df['title'], df['text'])
]

# ---------------------------------------------------------------------------
# Step 2. Embed texts
# ---------------------------------------------------------------------------
vecs = embed_texts(texts)  # returns float32 normalized vectors

# ---------------------------------------------------------------------------
# Step 3. Load or initialize FAISS index
# ---------------------------------------------------------------------------
index = load_index()
if index.ntotal == 0:
    # If index is empty, (re)create with the correct dimension
    index = create_index()

# ---------------------------------------------------------------------------
# Step 4. Add new vectors + metadata
# ---------------------------------------------------------------------------
add_vectors(index, ids, vecs, metas)

# (optional) Save index back to disk (if using file persistence)
# save_index(index)

print(f"Ingested {len(texts)} docs into FAISS index and SQLite metadata store.")
PY
