
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq


# --------------------------------
# LOAD GROQ API KEY
# --------------------------------

load_dotenv()

# First check Streamlit Cloud Secrets
api_key = os.getenv("GROQ_API_KEY")

try:
    api_key = st.secrets.get("GROQ_API_KEY", api_key)
except Exception:
    # Allows local execution using a .env file
    pass

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please configure it in Streamlit Secrets."
    )

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

    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No readable text found in the uploaded PDF.")

    embeddings = embedding_model.encode(chunks)

    # FAISS requires float32 embeddings
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(embeddings)

    stored_texts = chunks


# --------------------------------
# ASK QUESTION
# --------------------------------

def ask_question(question):
    global vector_store, stored_texts

    if vector_store is None or not stored_texts:
        return "Please upload a PDF first."

    query_embedding = embedding_model.encode([question])

    query_embedding = np.array(query_embedding).astype("float32")

    # Retrieve the two most relevant chunks
    k = min(2, len(stored_texts))

    D, I = vector_store.search(query_embedding, k=k)

    context_chunks = [
        stored_texts[i]
        for i in I[0]
        if 0 <= i < len(stored_texts)
    ]

    context = "\n".join(context_chunks)

    # Limit context length
    context = context[:2000]

    prompt = f"""
    Answer the question using ONLY the syllabus context below.

    If the answer is not available in the syllabus,
    clearly say that the information is not found in the uploaded syllabus.

    Context:
    {context}

    Question:
    {question}

    Answer in clear student-friendly language.
    """

    # Send question to Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
