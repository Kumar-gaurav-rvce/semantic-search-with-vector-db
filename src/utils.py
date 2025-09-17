# src/utils.py
"""
SQLite utilities for storing document metadata.

We keep FAISS embeddings on disk (faiss_index.bin) and store the mapping
id -> metadata (title, text, source, created_at) in a lightweight SQLite DB.
This allows fast lookup of metadata for search results.

Tables:
    docs (
      id TEXT PRIMARY KEY,
      title TEXT,
      text TEXT,
      source TEXT,
      created_at TEXT
    )
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

# Path to local SQLite DB
DB_PATH = Path("vector_index.db")


def get_conn() -> sqlite3.Connection:
    """
    Return a SQLite connection with row access by column name.

    Returns:
        sqlite3.Connection
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def ensure_metadata_table() -> None:
    """
    Ensure the docs table exists in the SQLite DB.
    If missing, create it.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
          id TEXT PRIMARY KEY,
          title TEXT,
          text TEXT,
          source TEXT,
          created_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def insert_or_update_doc(doc: Dict[str, Any]) -> None:
    """
    Insert or update a document's metadata.

    Args:
        doc: dict with keys id, title, text, source, created_at
    """
    ensure_metadata_table()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO docs (id, title, text, source, created_at)
        VALUES (:id, :title, :text, :source, :created_at)
        """,
        doc,
    )
    conn.commit()
    conn.close()


def fetch_doc(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single document's metadata by id.

    Args:
        doc_id: document identifier

    Returns:
        dict with metadata, or None if not found
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM docs WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_all_docs(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch multiple documents (useful for inspection/debug).

    Args:
        limit: maximum number of rows

    Returns:
        List of metadata dicts
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM docs LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
