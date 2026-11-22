
import os
import re
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

    # Remove unusual control characters from PDF extraction
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No readable text found in the PDF.")

    embeddings = embedding_model.encode(chunks)

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(embeddings)

    stored_texts = chunks

    return len(chunks)


# --------------------------------
# KEYWORD-BASED RETRIEVAL
# --------------------------------

def keyword_retrieval(question):
    global stored_texts

    query = question.lower()

    # Handle questions about a particular unit
    unit_match = re.search(r"\bunit\s+(\d+)\b", query)

    if unit_match:
        unit_number = unit_match.group(1)

        pattern = re.compile(
            rf"\bUNIT\s+{re.escape(unit_number)}\b",
            re.IGNORECASE
        )

        matching_indices = [
            i for i, chunk in enumerate(stored_texts)
            if pattern.search(chunk)
        ]

        if matching_indices:
            selected = set()

            # Include matching chunks and nearby chunks
            # so chapter details are not missed.
            for index in matching_indices:
                for nearby in range(
                    max(0, index - 1),
                    min(len(stored_texts), index + 4)
                ):
                    selected.add(nearby)

            return [
                stored_texts[i]
                for i in sorted(selected)
            ]

    # Search for exact topic phrases and important keywords
    cleaned_query = re.sub(r"[^a-z0-9\s]", " ", query)

    stop_words = {
        "what", "which", "where", "when", "how",
        "are", "is", "the", "in", "of", "does",
        "do", "explain", "tell", "me", "about",
        "discusses", "discuss", "covered", "include",
        "included", "find", "list", "give", "show",
        "chapter", "chapters", "unit", "topics"
    }

    keywords = [
        word for word in cleaned_query.split()
        if word not in stop_words and len(word) > 2
    ]

    if not keywords:
        return []

    scored_chunks = []

    for index, chunk in enumerate(stored_texts):
        normalized_chunk = re.sub(
            r"[^a-z0-9\s]",
            " ",
            chunk.lower()
        )

        score = sum(
            normalized_chunk.count(keyword)
            for keyword in keywords
        )

        if score > 0:
            scored_chunks.append((score, index))

    scored_chunks.sort(reverse=True)

    return [
        stored_texts[index]
        for score, index in scored_chunks[:5]
    ]


# --------------------------------
# ASK QUESTION
# --------------------------------

def ask_question(question):
    global vector_store, stored_texts

    if vector_store is None or not stored_texts:
        return "Please upload a PDF first."

    # First try keyword-based retrieval
    context_chunks = keyword_retrieval(question)

    # Use FAISS if keyword retrieval finds nothing
    if not context_chunks:

        query_embedding = embedding_model.encode([question])

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        k = min(5, len(stored_texts))

        distances, indices = vector_store.search(
            query_embedding,
            k=k
        )

        context_chunks = [
            stored_texts[index]
            for index in indices[0]
            if 0 <= index < len(stored_texts)
        ]

    context = "\n\n".join(context_chunks)

    # Show retrieved context during testing
    if st.session_state.get("debug_mode", False):
        st.write("DEBUG: Retrieved Context")
        st.text(context)

    prompt = f"""
You are SyllabAI, an academic syllabus assistant.

Answer the student's question using the syllabus context below.

IMPORTANT RULES:
1. Use only the provided syllabus information.
2. Never invent chapter names or unit numbers.
3. If the information is available, answer directly.
4. Preserve the original chapter names.
5. If information is genuinely absent, say so.
6. Use clear, student-friendly language.
7. When asked for chapters in a unit, list all matching chapters
   found in the provided context.

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
