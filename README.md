
# RAG Test - Simple Retrieval-Augmented Chatbot

This project is a simple, beginner-friendly RAG pipeline that uses local blog files as the knowledge base.
It includes:
- `blogs/` folder with sample `.txt` files
- `backend.py` that builds embeddings and a (FAISS) index, and exposes `get_answer(query)`
- `app.py` — a Streamlit app for interactive chatting
- `rag_pipeline.ipynb` — a Colab-friendly notebook showing the main steps
- `requirements.txt` — suggested dependencies

This project was prepared from the task description you uploaded. See the original task PDF included in the conversation for grading criteria and exact requirements.

## Quick start (local)

1. Make a python virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\\Scripts\\activate    # Windows (PowerShell)
```

2. Install dependencies (you may need to adjust `torch` and `faiss` for your platform/CUDA):
```bash
pip install -r requirements.txt
# If faiss-cpu fails on your platform, either use conda or remove faiss-cpu and rely on the numpy fallback in backend.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Quick start (Colab)
- Upload the project to Google Drive or GitHub, open `rag_pipeline.ipynb` in Colab, and run cells.
- In Colab you may want to `!pip install -q sentence-transformers faiss-cpu transformers torch` first.

## Notes & troubleshooting
- FAISS can be tricky to install on some platforms. If `faiss-cpu` fails, the app uses a numpy fallback (less performant but works).
- Downloading transformer models (e.g. gpt2) may take time and need sufficient disk space.
- If you want better answers, replace `gpt2` with a stronger, local model (if you have GPU and RAM) or use a hosted LLM via API (OpenAI, Hugging Face Inference, etc.).
