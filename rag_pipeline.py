
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq


# --------------------------------
# LOAD GROQ API KEY
# --------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

try:
    api_key = st.secrets.get("GROQ_API_KEY", api_key)
except Exception:
    pass

if not api_key:
    raise ValueError("GROQ_API_KEY is missing.")

client = Groq(api_key=api_key)


# --------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# VECTOR STORE
# --------------------------------

vector_store = None
stored_texts = []


# --------------------------------
# CREATE VECTOR STORE
# --------------------------------

def create_vector_store(text):
    global vector_store, stored_texts

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No readable text found in the PDF.")

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    vector_store = faiss.IndexFlatL2(dimension)

    vector_store.add(embeddings)

    stored_texts = chunks

    return len(chunks)


# --------------------------------
# ASK QUESTION
# --------------------------------

def ask_question(question):
    global vector_store, stored_texts

    if vector_store is None or not stored_texts:
        return "Please upload a PDF first."

    query_embedding = embedding_model.encode([question])

    query_embedding = np.array(query_embedding).astype("float32")

    # Retrieve five relevant chunks
    k = min(5, len(stored_texts))

    distances, indices = vector_store.search(
        query_embedding,
        k=k
    )

    context_chunks = []

    for index in indices[0]:
        if 0 <= index < len(stored_texts):
            context_chunks.append(stored_texts[index])

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are SyllabAI, an academic syllabus assistant.

Answer the student's question using the uploaded syllabus context.

IMPORTANT RULES:
1. Use only the information provided in the context.
2. Do not invent chapter names, units or topics.
3. If the requested information is present, answer it directly.
4. If the information is genuinely absent, say that it is not found.
5. Preserve the original chapter names and unit numbers.
6. Use clear, student-friendly language.

SYLLABUS CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
