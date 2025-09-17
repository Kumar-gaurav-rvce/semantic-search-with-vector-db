# semantic-search-with-vector-db

Demo project showing how to build a semantic search pipeline with a lightweight vector database.  
It demonstrates core concepts used in modern ML workflows:

- Compute text embeddings with [SentenceTransformers](https://www.sbert.net/)
- Store vectors in [FAISS](https://faiss.ai/) (similarity search)
- Store metadata in [SQLite](https://www.sqlite.org/)
- Run semantic search with [Streamlit](https://streamlit.io/)
- Add advanced features (query expansion, fuzzy matching, clustering, reranking, summarization, …)

---

## 🚀 Features

- **Vector DB**: FAISS index for nearest-neighbor search
- **Embeddings**: SentenceTransformers encoders
- **Metadata store**: SQLite (id, title, text)
- **Streamlit app** with:
  - Query expansion (WordNet synonyms)
  - Result highlighting
  - Filters and fuzzy matching
  - Re-ranking with cross-encoder
  - Clustering of results
  - Summarization of results
  - Top-K explanation chart
  - Result explorer with expandables
  - “Did you mean?” suggestions

---

## ⚡ Quickstart (local)

1. **Create a virtual environment & install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt


2. **Download NLTK data (required once)**
```bash
python - <<'PY'
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
PY

3. **(Optional) Pre-download cross-encoder model (faster first run)**
```bash
python - <<'PY'
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
PY

4. **Ingest sample data**
```bash
bash scripts/ingest_sample.sh

5. **Run the Streamlit app**
```bash
bash scripts/run_demo.sh

---

## Project Structure

├── app/
│   └── streamlit_app.py    # Streamlit UI
├── src/
│   ├── embed.py            # embedding model
│   ├── indexer.py          # FAISS index handling
│   ├── query.py            # main search pipeline
│   ├── utils.py            # SQLite utils
│   ├── expand.py           # query expansion
│   ├── rerank.py           # cross-encoder reranker
│   ├── clustering.py       # result clustering
│   └── summary.py          # extractive summarization
├── scripts/
│   ├── ingest_sample.sh    # ingest sample CSV into index + db
│   ├── run_demo.sh         # start Streamlit app
│   └── start_demo.sh       # optional wrapper with env vars
├── requirements.txt
└── README.md
