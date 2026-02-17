# RAG Chatbot - Homework 4

A Retrieval-Augmented Generation (RAG) chatbot built from scratch that ingests a knowledge base from PDF and audio sources, then answers questions using retrieved context.

## Architecture

```
User Question
      |
      v
 [LM Studio Embedding Model] ──> Query Vector
      |
      v
 [Qdrant Vector DB] ──> Top-10 Relevant Chunks
      |
      v
 [LM Studio Chat Model] ──> Generated Answer
```

### Pipeline Steps

1. **Load PDF** (`loader.py`) - Extracts text from PDF using `pypdf`
2. **Transcribe Audio** (`transcriber.py`) - Converts MP4 audio to text using OpenAI Whisper (`base` model)
3. **Chunk Text** (`chunker.py`) - Splits combined text into 800-character chunks with 150-character overlap using LangChain's `RecursiveCharacterTextSplitter`
4. **Embed** (`embeddings.py`) - Calls LM Studio's OpenAI-compatible `/v1/embeddings` endpoint for chunk and query embeddings with local embedding cache
5. **Store** (`vector_store.py`) - Stores embeddings in a persistent local Qdrant database (`data/cache/qdrant_db`) with cosine similarity; vector dimension is inferred from returned embeddings
6. **Retrieve & Generate** (`rag_pipeline.py`) - Queries the vector DB for the top-10 most relevant chunks and passes them as context to the LLM
7. **CLI Chatbot** (`main.py`) - Starts an interactive prompt so you can ask questions directly (`exit`, `quit`, `q` to stop)

## Project Structure

```
HomeWork4/
├── README.md
├── requirements.txt
├── data/
│   ├── cache/
│   │   ├── embedding_cache.json
│   │   ├── qdrant_db/
│   │   └── transcription.txt
│   ├── pdf/
│   │   └── Databases for GenAI.pdf
│   └── media/
│       └── 2 part Databases for GenAI.mp4
└── src/
    ├── main.py                # Entry point - orchestrates the full pipeline
    ├── loader.py              # PDF text extraction
    ├── transcriber.py         # Audio transcription with Whisper
    ├── chunker.py             # Text chunking
    ├── embeddings.py          # Vector embedding via LM Studio API
    ├── vector_store.py        # Qdrant vector database operations
    └── rag_pipeline.py        # LLM answer generation with retrieved context
```

## Setup & Installation

### Prerequisites

- Python 3.11
- [ffmpeg](https://ffmpeg.org/) (required by Whisper for audio processing)
- [LM Studio](https://lmstudio.ai/) with:
  - one embedding model loaded
  - one chat model loaded

### Install Dependencies

**IMPORTANT: Add required PDF to the `data/pdf/` directory and audio file to `data/media/`**

```bash
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu/Debian

pip install -r requirements.txt
```

### Optional Environment Variables

Both embeddings and chat use LM Studio's OpenAI-compatible server.

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_API_KEY="lm-studio"
export LM_STUDIO_EMBEDDING_MODEL="text-embedding-nomic-embed-text-v1.5"
export LM_STUDIO_CHAT_MODEL="google/gemma-3-4b"
```

### Start LM Studio Server

1. Open LM Studio
2. Load an embedding model and a chat model
3. Go to the **Developer** tab and click **Start Server** (runs on `localhost:1234`)

### Run the Pipeline

```bash
python src/main.py
```

The pipeline will:
- Extract text from the PDF
- Transcribe the audio file (takes several minutes on CPU)
- Chunk, embed, and store in the persistent vector database
- Start interactive CLI chat mode

### CLI Usage

After startup, ask questions directly:

```text
Ask something: Why is hybrid search better than vector-only search?
Assistant: ...
Ask something: exit
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | pypdf |
| Speech-to-Text | OpenAI Whisper (base) |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | LM Studio Embeddings API (`/v1/embeddings`) |
| Embedding Cache | JSON file in `data/cache/embedding_cache.json` |
| Vector Database | Qdrant (persistent local storage) |
| LLM | Chat model via LM Studio (`/v1/chat/completions`) |
