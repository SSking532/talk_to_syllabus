import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# Load API key
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vector_store = None
stored_texts = []


def create_vector_store(text):
    global vector_store, stored_texts

    # Smaller chunk size to avoid token overflow
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(text)

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]
    vector_store = faiss.IndexFlatL2(dimension)
    vector_store.add(np.array(embeddings))

    stored_texts = chunks


def ask_question(question):
    global vector_store, stored_texts

    if vector_store is None:
        return "Please upload a PDF first."

    query_embedding = embedding_model.encode([question])

    # Reduced retrieval size
    D, I = vector_store.search(np.array(query_embedding), k=2)

    context = "\n".join([stored_texts[i] for i in I[0]])

    # Limit context length
    context = context[:2000]

    prompt = f"""
    Answer the question using ONLY the syllabus context below.

    Context:
    {context}

    Question:
    {question}

    Answer in clear student-friendly language.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
