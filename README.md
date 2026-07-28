# 🔬 AI Research & Knowledge Assistant

A production-ready REST API for intelligent document management, semantic search, RAG-powered question answering with citations, document comparison, summarization, and TensorFlow document classification — built with **FastAPI**, **Google Gemini**, and **FAISS**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                      │
├──────────┬──────────┬───────────┬────────┬─────────────┤
│ /docs    │ /search  │/assistant │  /ml   │ /analytics  │
│ upload   │ semantic │   ask     │ train  │   stats     │
│ list     │ keyword  │ summarize │classify│             │
│ delete   │ hybrid   │ compare   │        │             │
└────┬─────┴────┬─────┴─────┬─────┴───┬────┴─────────────┘
     │          │           │         │
     ▼          ▼           ▼         ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│Document │ │ FAISS  │ │Gemini  │ │ TensorFlow   │
│  Store  │ │ Index  │ │  LLM   │ │  Classifier  │
│(JSON DB)│ │(Vector)│ │  API   │ │  (.keras)    │
└─────────┘ └────────┘ └────────┘ └──────────────┘
```

---

## 📁 Project Structure

```
ai-research-assistant/
│
├── main.py                          # FastAPI app entry point
├── requirements.txt
├── .env.example
│
├── app/
│   ├── api/routes/
│   │   ├── documents.py             # Upload, list, delete, reprocess
│   │   ├── search.py                # Semantic, keyword, hybrid search
│   │   ├── assistant.py             # QA, summarize, compare, sessions
│   │   ├── ml.py                    # Train, classify, predict
│   │   └── analytics.py             # Usage statistics
│   │
│   ├── core/
│   │   ├── config.py                # Centralised settings from .env
│   │   └── logging.py               # Structured logging
│   │
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   │
│   ├── services/
│   │   ├── document_store.py        # JSON-backed metadata persistence
│   │   ├── rag_pipeline.py          # Extract → chunk → embed → FAISS index
│   │   ├── ai_service.py            # Gemini LLM: QA, summary, comparison
│   │   ├── conversation_memory.py   # Per-session conversation history
│   │   └── analytics.py             # Query tracking & usage metrics
│   │
│   └── ml/
│       └── classifier.py            # TensorFlow model: train, save, predict
│
├── data/
│   ├── uploads/                     # Stored PDF files
│   ├── indexes/                     # FAISS indexes per document
│   └── models/                      # Saved TF model + tokenizer
│
├── tests/
│   └── test_api.py                  # Integration tests
│
└── sample_docs/                     # Sample PDFs for testing
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 2. Install
```bash
git clone https://github.com/YOUR_USERNAME/ai-research-assistant.git
cd ai-research-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY
```

### 4. Run
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open API Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `GEMINI_MODEL` | `models/gemini-2.5-flash` | LLM model for generation |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `CHUNK_SIZE` | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MAX_FILE_SIZE_MB` | `50` | Maximum PDF upload size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 📡 API Documentation

### Document Management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a PDF document |
| `GET` | `/documents/` | List all documents |
| `GET` | `/documents/{doc_id}` | Get document details |
| `DELETE` | `/documents/{doc_id}` | Delete a document |
| `POST` | `/documents/{doc_id}/reprocess` | Reprocess a document |

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search/` | Search with semantic/keyword/hybrid mode |

### AI Assistant

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/assistant/ask` | Ask a question (with conversation memory) |
| `POST` | `/assistant/summarize` | Summarize a document |
| `POST` | `/assistant/compare` | Compare multiple documents |
| `GET` | `/assistant/session/{id}` | Get conversation history |
| `DELETE` | `/assistant/session/{id}` | Clear conversation history |

### ML Classification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ml/train` | Train TensorFlow classifier |
| `GET` | `/ml/status` | Check training status |
| `POST` | `/ml/classify/{doc_id}` | Classify an uploaded document |
| `POST` | `/ml/classify-text` | Classify raw text |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/analytics/` | Get usage statistics |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔧 Design Decisions

### Chunking Strategy
- **RecursiveCharacterTextSplitter** with `chunk_size=800`, `chunk_overlap=100`
- Splits on paragraph → sentence → word boundaries to preserve semantic coherence
- Overlap prevents critical context being split across chunk boundaries

### Vector Search
- **FAISS IndexFlatIP** with L2-normalized vectors = cosine similarity
- One FAISS index per document for efficient per-document and cross-document search
- Hybrid search combines semantic (70%) + keyword (30%) scores

### Search Mode Selection
- **Semantic**: Best for conceptual queries ("explain the methodology")
- **Keyword**: Best for exact term lookup ("find mentions of BERT")
- **Hybrid**: Best general-purpose option

### TensorFlow Classifier
- Embedding → GlobalAveragePooling → Dense(128) → Dense(64) → Softmax
- Trained on domain-representative synthetic data per category
- Automatically classifies documents on upload once trained

### Conversation Memory
- In-memory per-session store (last 20 turns)
- Last 6 turns injected into LLM context for multi-turn QA

---

## ⚠️ Limitations
- Free Gemini API tier has rate limits — large documents may slow indexing
- Conversation memory resets on server restart (no persistent session DB)
- TF classifier trained on synthetic data — performance improves with real labeled data
- PDF text extraction may struggle with scanned/image-based PDFs

---

## 🚀 Future Improvements
- PostgreSQL/SQLite for persistent document metadata
- Redis for session memory persistence
- OCR support for scanned PDFs (Tesseract)
- Streaming LLM responses via Server-Sent Events
- Docker + docker-compose for containerized deployment
- BM25 + vector hybrid retrieval with reranking
- Multi-user authentication with JWT
