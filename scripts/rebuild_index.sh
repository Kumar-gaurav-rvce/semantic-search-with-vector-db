# locally, where you can run embedding backend (set OPENAI_API_KEY if using OpenAI)
export OPENAI_API_KEY="sk-xxxxxx"   # only if you use OpenAI embeddings
python - <<'PY'
import pandas as pd, numpy as np
from src.embed import embed_texts
from src.indexer import create_index, add_vectors, INDEX_PATH, DOC_IDS_PATH
from src.utils import ensure_metadata_table
from pathlib import Path

# ensure sample data exists
df = pd.read_csv("data/sample_docs.csv")
texts = df['text'].astype(str).tolist()
ids = df['id'].astype(str).tolist()
metas = [{"title": t, "text": txt, "source":"sample", "created_at": ""} for t,txt in zip(df['title'], df['text'])]

vecs = embed_texts(texts)
print("Embedding shape:", vecs.shape)
dim = vecs.shape[1]
idx = create_index(dim=dim)
# optional dedupe: keep first occurrence of each id
seen = set(); u_ids=[]; u_vecs=[]; u_metas=[]
for i,v,m in zip(ids, vecs, metas):
    if i in seen: continue
    seen.add(i); u_ids.append(i); u_vecs.append(v); u_metas.append(m)
u_vecs = np.stack(u_vecs).astype("float32")
add_vectors(idx, u_ids, u_vecs, u_metas)
print("Wrote index:", INDEX_PATH, "ntotal:", idx.ntotal)
PY
