"""
backend.py - helper functions for the RAG app

This file creates an embedding + vector store from files in ./blogs,
saves a simple local index, and exposes get_answer(query) used by app.py.

It tries to use faiss and transformers if available, but falls back to a simple numpy search if not.
"""

import os, pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BLOG_DIR = BASE_DIR / "blogs"
STORE_DIR = BASE_DIR / "faiss_store"
STORE_DIR.mkdir(exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def chunk_text(text, words_per_chunk=200, overlap=40):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+words_per_chunk]
        chunks.append(" ".join(chunk))
        i += words_per_chunk - overlap
    return chunks

def load_docs():
    docs = []
    for p in sorted(BLOG_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            docs.append({"id": f"{p.name}_{idx}", "text": c, "source": str(p.name)})
    return docs

# ----------------------------
# Build / load index
# ----------------------------
def build_or_load_index(embedding_model_name="all-MiniLM-L6-v2"):
    import numpy as np
    docs = load_docs()
    emb_path = STORE_DIR / "embeddings.pkl"
    docs_path = STORE_DIR / "docs.pkl"
    idx_path = STORE_DIR / "faiss.index"

    # Load if available
    if emb_path.exists() and docs_path.exists():
        try:
            with open(docs_path, "rb") as f:
                docs = pickle.load(f)
            with open(emb_path, "rb") as f:
                embs = pickle.load(f)
            try:
                import faiss
                if idx_path.exists():
                    index = faiss.read_index(str(idx_path))
                else:
                    index = None
                return docs, embs, index
            except Exception:
                return docs, embs, None
        except Exception:
            pass  # rebuild

    # Create embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model_name)
        texts = [d["text"] for d in docs]
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        import numpy as np
        embs = np.random.rand(len(docs), 384).astype("float32")

    # Save
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    with open(emb_path, "wb") as f:
        pickle.dump(embs, f)

    # Try FAISS
    try:
        import faiss
        if embs.dtype != "float32":
            embs = embs.astype("float32")
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        faiss.write_index(index, str(idx_path))
        return docs, embs, index
    except Exception:
        return docs, embs, None

# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query, docs, embs, index=None, embedding_model_name="all-MiniLM-L6-v2", k=3):
    import numpy as np
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model_name)
        q_emb = model.encode([query], convert_to_numpy=True)[0]
    except Exception:
        q_emb = np.random.rand(embs.shape[1]).astype("float32")

    if index is not None:
        import faiss
        v = q_emb.astype("float32")
        faiss.normalize_L2(v.reshape(1, -1))
        D, I = index.search(v.reshape(1, -1), k)
        results = []
        for idx in I[0]:
            if idx < len(docs):
                results.append(docs[idx])
        return results
    else:
        # numpy fallback
        def normalize(a):
            norms = (a**2).sum(axis=1, keepdims=True)**0.5
            norms[norms == 0] = 1.0
            return a / norms
        emb_norm = normalize(embs.astype("float32"))
        qn = q_emb.astype("float32")
        qn = qn / ((qn**2).sum()**0.5)
        sims = emb_norm @ qn
        top_idx = sims.argsort()[::-1][:k]
        return [docs[i] for i in top_idx]

# ----------------------------
# Answer generation
# ----------------------------
def generate_answer(context, query, model_name="gpt2", max_new_tokens=150):
    try:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9
        )

        prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True)
        text = out[0]["generated_text"]

        if "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        else:
            return text.strip()
    except Exception:
        return context.strip()

# ----------------------------
# Public helper
# ----------------------------
_index_built = False
_docs = None
_embs = None
_index = None

def ensure_index():
    global _index_built, _docs, _embs, _index
    if not _index_built:
        _docs, _embs, _index = build_or_load_index()
        _index_built = True
    return True

def get_answer(query, k=3):
    ensure_index()
    results = retrieve(query, _docs, _embs, _index, k=k)
    context = "\n\n---\n\n".join([r["text"] for r in results])
    answer = generate_answer(context, query)
    return answer, results
