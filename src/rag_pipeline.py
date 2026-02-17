import os

from embeddings import embed_query
from vector_store import search
from openai import OpenAI

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", "google/gemma-3-4b")

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)

def generate_answer(query: str) -> str:
    query_vector = embed_query(query)
    context_chunks = search(query_vector, top_k=10)

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful RAG assistant. Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "This is not covered in the provided material."

Context:
{context}

Question:
{query}"""

    response = client.chat.completions.create(
        model=LM_STUDIO_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content or ""
