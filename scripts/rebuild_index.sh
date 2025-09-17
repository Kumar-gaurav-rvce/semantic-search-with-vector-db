#!/usr/bin/env bash
set -euo pipefail

# Usage:
# export OPENAI_API_KEY="sk-..."
# bash scripts/rebuild_index_force_openai.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Using Python from: $(which python)"
python - <<'PY'
import os, sys, math
import pandas as pd
import numpy as np
from pathlib import Path
from src.indexer import create_index, add_vectors, INDEX_PATH, DOC_IDS_PATH
from src.utils import ensure_metadata_table

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("OPENAI_API_KEY not set in environment. Aborting.", file=sys.stderr)
    sys.exit(2)

# prefer new OpenAI SDK if available
use_new = False
try:
    import openai
    if hasattr(openai, "OpenAI"):  # newer SDK
        use_new = True
except Exception:
    print("openai package not available. Please pip install openai.", file=sys.stderr)
    sys.exit(2)

def openai_embed(texts, model="text-embedding-3-small"):
    # chunk in batches
    all_embs = []
    if use_new:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        for i in range(0, len(texts), 16):
            chunk = texts[i:i+16]
            resp = client.embeddings.create(model=model, input=chunk)
            # resp.data is list of embeddings
            for item in resp.data:
                all_embs.append(item.embedding)
    else:
        import openai
        openai.api_key = OPENAI_KEY
        for i in range(0, len(texts), 16):
            chunk = texts[i:i+16]
            resp = openai.Embedding.create(model=model, input=chunk)
            for item in resp["data"]:
                all_embs.append(item["embedding"])
    arr = np.asarray(all_embs, dtype="float32")
    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    arr = arr / norms
    return arr

# load data
DATA_CSV = Path("data/sample_docs.csv")
if not DATA_CSV.exists():
    print("Missing data/sample_docs.csv", file=sys.stderr)
    sys.exit(2)

df = pd.read_csv(DATA_CSV)
texts = df['text'].astype(str).tolist()
ids = df['id'].astype(str).tolist()
metas = [{"title": t, "text": txt, "source":"sample", "created_at": ""} for t,txt in zip(df['title'], df['text'])]

print("Requesting OpenAI embeddings for", len(texts), "documents. This will use billing.")
vecs = openai_embed(texts)   # shape (N, dim)
print("Embedding shape:", vecs.shape)

dim = vecs.shape[1]
print("Creating FAISS index with dim:", dim)
idx = create_index(dim=dim)

# dedupe by id (first occurrence)
seen = set(); u_ids=[]; u_vecs=[]; u_metas=[]
for i,v,m in zip(ids, vecs, metas):
    if i in seen:
        continue
    seen.add(i)
    u_ids.append(i); u_vecs.append(v); u_metas.append(m)

u_vecs = np.stack(u_vecs).astype("float32")
add_vectors(idx, u_ids, u_vecs, u_metas)
print("Wrote index:", Path(INDEX_PATH).resolve(), "ntotal:", idx.ntotal)
PY
