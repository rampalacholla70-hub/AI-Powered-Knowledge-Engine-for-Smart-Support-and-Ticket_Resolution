from pathlib import Path
import os
import sys
import re
from docx import Document as DocxDocument
from chromadb import Client
from chromadb.utils import embedding_functions
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(BASE_DIR, "docs", "dell-data")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if not GROQ_KEY and not GEMINI_KEY:
    print(" ERROR: No API keys found in backend file.")
    sys.exit(1)

try:
    groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
except Exception as e:
    groq_client = None
    print(f" Groq init failed: {e}")

try:
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    else:
        gemini_model = None
except Exception as e:
    gemini_model = None
    print(f" Gemini init failed: {e}")

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def read_docx(filepath):
    doc = DocxDocument(filepath)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

def semantic_chunk_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    chunks = re.split(r'\n\s*\n', text)
    return [c.strip() for c in chunks if c.strip()]

def create_vectorstore():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise FileNotFoundError(f"No .docx files found in {DOCS_DIR}")

    docs = []
    for file in sorted(os.listdir(DOCS_DIR)):
        if file.lower().endswith(".docx"):
            filepath = os.path.join(DOCS_DIR, file)
            text = read_docx(filepath)
            if not text:
                continue
            chunks = semantic_chunk_text(text)
            for idx, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={"source": file, "chunk": idx}))
            print(f" {len(chunks)} chunks created from {file}")

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    chroma = Client()

    vectordb = chroma.create_collection(
        name="dell_kb",
        embedding_function=embedding_func
    )

    vectordb.add(
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        ids=[str(i) for i in range(len(docs))]
    )

    print(f"\n Vectorstore created with {len(docs)} semantic chunks\n")
    return vectordb

def load_vectorstore():
    chroma = Client()
    try:
        return chroma.get_collection(name="dell_kb")
    except Exception:
        print(" Vectorstore not found → Creating new one...")
        return create_vectorstore()

def get_ai_response(query, docs):
    context = "\n\n".join(docs)

    prompt = f"""
You are a Dell technical support assistant.
Use the following Dell knowledge base documentation to answer the question.

Context:
{context}

Question: {query}

Give a helpful and accurate answer.
"""

    # Groq
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(" Groq failed:", e)

    # Gemini
    if gemini_model:
        try:
            resp = gemini_model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            print(" Gemini failed:", e)

    return "No LLM available."

# a single callable function the frontend will import
global_vectordb = None

def ensure_vectordb():
    global global_vectordb
    if global_vectordb is None:
        global_vectordb = load_vectorstore()
    return global_vectordb

def run_rag(query: str, n_results: int = 4):
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    vectordb = ensure_vectordb()
    results = vectordb.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0] if results and results.get("documents") else []

    if not docs:
        ai_response = " I couldn't find any relevant information in the KB for your query."
    else:
        ai_response = get_ai_response(query, docs)

    response_data = {
        "user_query": query,
        "rag_response": ai_response,
        "doc_context": docs
    }
    return response_data

if __name__ == "__main__":
    print("Test run (no Flask). This will build or load the vectorstore and run a sample query.")
    vect = ensure_vectordb()
    sample = run_rag("My laptop won't turn on", n_results=2)
    print(sample)
