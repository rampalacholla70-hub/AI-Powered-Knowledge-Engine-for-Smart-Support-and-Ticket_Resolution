# utils/dell_knowledge_query_groq.py
from pathlib import Path
import os
import sys
import re
from docx import Document as DocxDocument
from chromadb import Client
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import traceback

# Optional imports (wrapped so module can load even if these aren't installed)
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# Try importing Streamlit for secrets support (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

# Load environment variables from .env for local dev
load_dotenv()

# Helper to fetch API key: prefer Streamlit secrets, support nested keys, then env var.
def get_api_key(key_name):
    # Try streamlit secrets in several common shapes
    if STREAMLIT_AVAILABLE:
        try:
            # direct key
            if key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass
        try:
            # nested style: st.secrets["GROQ"]["API_KEY"] or st.secrets["GROQ_API"]["key"]
            # check common groupings
            for group in ("GROQ", "GROQ_API", "GROQ_API_KEY", "GROQ_KEYS"):
                try:
                    g = st.secrets.get(group, None)
                    if isinstance(g, dict):
                        # prefer fields named "API_KEY" / "api_key" / "key"
                        for k in ("API_KEY", "api_key", "key"):
                            if k in g and g[k]:
                                return g[k]
                except Exception:
                    pass
            # also try top-level names like st.secrets["GEMINI_API_KEY"]
            if key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass

    # Fallback to environment variable
    val = os.getenv(key_name)
    if val:
        return val

    # Try alternate env var names (common)
    alt_map = {
        "GROQ_API_KEY": ["GROQ_KEY", "GROQAPIKEY", "GROQ"],
        "GEMINI_API_KEY": ["GEMINI_KEY", "GEMINIAPIKEY", "GEMINI"]
    }
    for alt in alt_map.get(key_name, []):
        v = os.getenv(alt)
        if v:
            return v
    return None


# Acquire keys
GROQ_KEY = get_api_key("GROQ_API_KEY")
GEMINI_KEY = get_api_key("GEMINI_API_KEY")

# Tokenizers parallelism (avoid noisy HF warnings)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(BASE_DIR, "docs", "dell-data")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 800))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", 200))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", 200))

# Helpful startup logs
if not GROQ_KEY and not GEMINI_KEY:
    print("WARNING: No LLM API keys (Groq/Gemini) found in secrets or environment. LLM responses will be unavailable unless configured.")

# Initialize LLM clients (wrapped with safety)
groq_client = None
gemini_model = None

if GROQ_KEY and Groq is not None:
    try:
        groq_client = Groq(api_key=GROQ_KEY)
        print("Groq client initialized.")
    except Exception as e:
        groq_client = None
        print("Groq init failed:", e)
        traceback.print_exc()
else:
    if GROQ_KEY is None:
        print("Groq API key not found.")
    else:
        print("Groq package not installed or import failed; Groq unavailable.")

if GEMINI_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_KEY)
        # choose a reasonably available model name; fallback handled later
        try:
            gemini_model = genai.GenerativeModel("gemini-2.5-pro")
        except Exception:
            # some deployments require default usage; keep object None if cannot init
            gemini_model = None
        print("Gemini configured.")
    except Exception as e:
        gemini_model = None
        print("Gemini init failed:", e)
        traceback.print_exc()
else:
    if GEMINI_KEY is None:
        print("Gemini API key not found.")
    else:
        print("google.generativeai not available; Gemini client unavailable.")


# Optional: LangChain SemanticChunker (not required)
SEMANTIC_CHUNKER_AVAILABLE = False
try:
    from langchain_experimental.text_splitter import SemanticChunker
    # langchain_huggingface package (if present)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None
    SEMANTIC_CHUNKER_AVAILABLE = True
except Exception:
    SEMANTIC_CHUNKER_AVAILABLE = False


# Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


# docx reader
def read_docx(filepath):
    doc = DocxDocument(filepath)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])


# Simple robust sentence splitter
def _split_sentences(text: str):
    text = text.strip()
    sentence_end_re = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"“])')
    parts = sentence_end_re.split(text)
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > CHUNK_MAX_CHARS * 2:
            subparts = re.split(r',\s+', p)
            for s in subparts:
                s = s.strip()
                if s:
                    result.append(s)
        else:
            result.append(p)
    return result


def chunk_text_to_list(text: str):
    if not text or not text.strip():
        return []

    # Try SemanticChunker if available and embeddings present
    if SEMANTIC_CHUNKER_AVAILABLE and HuggingFaceEmbeddings is not None:
        try:
            hf_embed = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            splitter = SemanticChunker(hf_embed, breakpoint_threshold_type="percentile")
            docs = splitter.create_documents([text])
            out_chunks = []
            for d in docs:
                if hasattr(d, "page_content"):
                    out_chunks.append(d.page_content.strip())
                elif isinstance(d, str):
                    out_chunks.append(d.strip())
                else:
                    out_chunks.append(str(d).strip())
            if out_chunks:
                print(f"Semantic chunking successful: {len(out_chunks)} chunks")
                return out_chunks
        except Exception as e:
            print(f"SemanticChunker failed (falling back): {e}")
            traceback.print_exc()

    # Fallback sentence-accumulation chunker
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    paragraphs = re.split(r'\n\s*\n', text)

    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.splitlines()
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if re.match(r'^\#{1,6}\s+', ln) or (len(ln) <= 80 and ln.isupper()):
                sentences.append(ln)
            else:
                sentences.extend(_split_sentences(ln))

    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        s_len = len(s)
        if s_len > CHUNK_MAX_CHARS:
            parts = re.split(r'([,;])\s*', s)
            buffer = ""
            for p in parts:
                if not p:
                    continue
                if len(buffer) + len(p) + 1 > CHUNK_MAX_CHARS and buffer:
                    current.append(buffer.strip())
                    current_len += len(buffer)
                    buffer = p
                else:
                    buffer = buffer + " " + p if buffer else p
            if buffer:
                current.append(buffer.strip())
                current_len += len(buffer)
        else:
            current.append(s)
            current_len += s_len

        if current_len >= CHUNK_MAX_CHARS:
            chunk_text = " ".join(current).strip()
            chunks.append(chunk_text)
            if CHUNK_OVERLAP_CHARS > 0:
                overlap_buf = []
                overlap_len = 0
                for cs in reversed(current):
                    if overlap_len >= CHUNK_OVERLAP_CHARS:
                        break
                    overlap_buf.insert(0, cs)
                    overlap_len += len(cs)
                current = overlap_buf.copy()
                current_len = sum(len(x) for x in current)
            else:
                current = []
                current_len = 0

    if current:
        tail = " ".join(current).strip()
        if len(tail) < CHUNK_MIN_CHARS and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + tail
        else:
            chunks.append(tail)

    final_chunks = [c.strip() for c in chunks if c and len(c.strip()) >= 1]
    return final_chunks


# Vectorstore create/load helpers
def create_vectorstore():
    """
    Build a Chroma collection from DOCX files in DOCS_DIR.
    Raises FileNotFoundError if no .docx files present.
    """
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        raise FileNotFoundError(f"No .docx files found in {DOCS_DIR}")

    docs = []
    for file in sorted(os.listdir(DOCS_DIR)):
        if file.lower().endswith(".docx"):
            filepath = os.path.join(DOCS_DIR, file)
            text = read_docx(filepath)
            if not text:
                continue
            chunks = chunk_text_to_list(text)
            for idx, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={"source": file, "chunk": idx}))
            print(f"{len(chunks)} chunks created from {file}")

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    chroma = Client()

    # attempt to delete existing to rebuild cleanly
    try:
        existing = chroma.get_collection(name="dell_kb")
        if existing:
            try:
                chroma.delete_collection("dell_kb")
                print("Existing 'dell_kb' collection deleted to rebuild.")
            except Exception:
                pass
    except Exception:
        pass

    vectordb = chroma.create_collection(name="dell_kb", embedding_function=embedding_func)
    vectordb.add(
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        ids=[str(i) for i in range(len(docs))]
    )

    print(f"Vectorstore created with {len(docs)} semantic chunks")
    return vectordb


def load_vectorstore():
    chroma = Client()
    try:
        return chroma.get_collection(name="dell_kb")
    except Exception:
        print("Vectorstore not found → Creating new one...")
        return create_vectorstore()


# Dummy vectorstore wrapper used when real vectorstore unavailable:
class DummyVectorDB:
    def query(self, query_texts=None, n_results=4, **kwargs):
        # returns same shape as chroma query results but empty
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}


# AI response: calls Groq or Gemini (if available)
def get_ai_response(query, docs):
    context = "\n\n".join(docs) if docs else ""

    prompt = f"""
You are a Dell technical support assistant.

IMPORTANT RULE:
You ONLY answer questions related to Dell laptops, Dell devices, Dell troubleshooting, warranty, drivers, configurations, and Dell services.
If the user asks anything personal, unrelated, or non-technical, respond with: 
"I'm here to help only with Dell-related queries. Please ask a Dell-related question."

Use the following Dell knowledge base documentation to answer the question.

Context:
{context}

User Question: {query}

Give a helpful and accurate answer ONLY if the question is Dell-related.
"""

    # Prefer Groq if available
    if groq_client:
        try:
            # Wrap in try/except to surface errors
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            # safe path to get text
            try:
                return response.choices[0].message.content.strip()
            except Exception:
                # fallback: try raw str
                return str(response)
        except Exception as e:
            print("Groq call failed:", e)
            traceback.print_exc()

    # Next: Gemini
    if gemini_model:
        try:
            resp = gemini_model.generate_content(prompt)
            # Gemini returns resp.text in many wrappers
            return getattr(resp, "text", str(resp)).strip()
        except Exception as e:
            print("Gemini call failed:", e)
            traceback.print_exc()

    # If neither available, clearly inform the caller
    return "RAG backend (LLM) not configured or failed to respond. Please ensure GROQ_API_KEY or GEMINI_API_KEY is set."


# global cache for vectordb
global_vectordb = None


def ensure_vectordb():
    """
    Try to return a live vectordb. If vectorstore cannot be loaded (missing files or errors),
    return a DummyVectorDB so callers can continue without crashes.
    """
    global global_vectordb
    if global_vectordb is None:
        try:
            # try to load existing collection
            global_vectordb = load_vectorstore()
            # If collection loaded but appears empty, warn and keep it (query will return empty docs)
            try:
                # many chroma clients expose .count() or length via get_collection().get()['ids']
                # we won't rely on a specific API — best-effort check:
                res = global_vectordb.query(query_texts=["ping"], n_results=1)
                if not res or not res.get("documents") or not any(res["documents"]):
                    print("Vectorstore appears empty (no docs indexed).")
            except Exception:
                # ignore
                pass
        except FileNotFoundError as fe:
            print("create_vectorstore failed: no docs found:", fe)
            global_vectordb = DummyVectorDB()
        except Exception as e:
            print("Error loading vectorstore:", e)
            traceback.print_exc()
            global_vectordb = DummyVectorDB()
    return global_vectordb


def run_rag(query: str, n_results: int = 4):
    """
    Main callable used by frontend. Always returns dict:
      { "user_query": str, "rag_response": str, "doc_context": list }
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    try:
        vectordb = ensure_vectordb()
        # guard vectordb.query errors
        try:
            results = vectordb.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            print("Vectorstore query failed:", e)
            traceback.print_exc()
            results = {"documents": [[]]}

        # Extract docs safely
        docs = []
        if results and isinstance(results, dict):
            if results.get("documents"):
                # support list-of-lists shape
                docs = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"][0]
            elif results.get("matches"):
                # some clients return matches list
                docs = [m.get("document") or m.get("content") for m in results.get("matches", [])][:n_results]
        # fallback ensure list
        docs = docs or []

        # If no docs were retrieved, return message but still attempt to call LLM if desired
        if not docs:
            # We still try an LLM answer with empty context — useful if KB missing but LLM available
            ai_response = get_ai_response(query, docs)
            # If LLM unavailable this will be a clear string
            response_data = {
                "user_query": query,
                "rag_response": ai_response,
                "doc_context": []
            }
            return response_data

        # If docs found, call LLM with the retrieved docs
        ai_response = get_ai_response(query, docs)
        response_data = {
            "user_query": query,
            "rag_response": ai_response,
            "doc_context": docs
        }
        return response_data

    except Exception as e:
        print("run_rag unexpected error:", e)
        traceback.print_exc()
        return {
            "user_query": query,
            "rag_response": f"RAG runtime error: {e}",
            "doc_context": []
        }


# allow quick CLI test
if __name__ == "__main__":
    print("Test run: build/load vectorstore and run sample query.")
    try:
        vect = ensure_vectordb()
        sample = run_rag("My laptop won't turn on", n_results=2)
        print(sample)
    except Exception as e:
        print("Self-test failed:", e)
        traceback.print_exc()
