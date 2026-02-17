from pathlib import Path

from loader import load_pdf
from transcriber import transcribe_audio
from chunker import chunk_text
from embeddings import embed_texts
from vector_store import init_collection, insert_chunks, close_client
from rag_pipeline import generate_answer

BASE_DIR = Path(__file__).resolve().parent.parent


def run_cli_chatbot():
    print("\n" + "=" * 60)
    print("RAG CHATBOT - CLI MODE")
    print("Type a question and press Enter.")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 60)

    while True:
        question = input("\nAsk something: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        answer = generate_answer(question)
        print(f"\nAssistant:\n{answer}")


def main():
    pdf_path = BASE_DIR / "data" / "pdf" / "Databases for GenAI.pdf"
    audio_path = BASE_DIR / "data" / "media" / "2 part Databases for GenAI.mp4"

    try:
        # --- Load PDF ---
        print("[1/5] Loading PDF...")
        pdf_text = load_pdf(str(pdf_path))
        print(f"       PDF extracted: {len(pdf_text)} characters")

        # --- Transcribe audio ---
        print("[2/5] Transcribing audio (this may take a while)...")
        audio_text = transcribe_audio(str(audio_path))
        print(f"       Transcription done: {len(audio_text)} characters")

        # --- Chunk with source tracking ---
        print("[3/5] Chunking text...")
        pdf_chunks = chunk_text(pdf_text)
        audio_chunks = chunk_text(audio_text)

        chunks = pdf_chunks + audio_chunks
        metadatas = (
            [{"source": "PDF", "chunk_index": i} for i in range(len(pdf_chunks))]
            + [{"source": "Audio Transcript", "chunk_index": i} for i in range(len(audio_chunks))]
        )
        print(f"       Created {len(chunks)} chunks ({len(pdf_chunks)} PDF, {len(audio_chunks)} audio)")

        # --- Embed ---
        print("[4/5] Embedding chunks...")
        embeddings = embed_texts(chunks)
        print(f"       Embedded {len(embeddings)} chunks")
        vector_size = len(embeddings[0]) if embeddings else 384

        # --- Store ---
        print("[5/5] Storing in Qdrant vector database...")
        init_collection(vector_size=vector_size)
        insert_chunks(chunks, embeddings, metadatas=metadatas)
        print("       Vector store ready")

        run_cli_chatbot()
    finally:
        close_client()


if __name__ == "__main__":
    main()
