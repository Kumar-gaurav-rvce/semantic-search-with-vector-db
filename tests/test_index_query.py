def test_embed_index_search():
    # smoke test: make sure embeddings and index exist after ingest
    import src.embed as emb
    import src.indexer as idx
    from src.utils import ensure_metadata_table
    ensure_metadata_table()
    model = emb.get_model()
    v = emb.embed_texts(["hello world"])
    assert v.shape[1] == 384
    # index creation
    index = idx.create_index()
    idx.save_index(index)
    loaded = idx.load_index()
    assert loaded.ntotal == 0
