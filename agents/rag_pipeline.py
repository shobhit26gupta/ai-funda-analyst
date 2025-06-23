import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import os

import pytesseract

# Example path (adjust if yours is different)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


if not os.path.exists("static"):
    os.makedirs("static")
# Global storage (for demo purposes; consider more robust session/state management in production)
texts = []
index = None
model = None
llm = None

# --- Extract Text from PDF (OCR fallback) ---
def extract_text_from_pdf(file_path: str) -> str:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(file_path)
    all_text = ""
    for page_num, page in enumerate(doc):
        # Try to extract text
        text = page.get_text()
        if not text.strip():
            # No text found — fallback to OCR
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")  # Render image in memory
            img = Image.open(io.BytesIO(img_bytes))

            # Perform OCR with no output config
            text = pytesseract.image_to_string(img, config='')  # no output folder

        all_text += text + "\n"
    return all_text
# --- Split & Embed Text ---
def embed_text_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    global texts, index, model
    texts = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        texts.append(chunk)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

# --- Setup LangChain LLM ---
def setup_agent():
    global llm
    openai_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=openai_key, model_name="gpt-4.1-nano", temperature=0)

# --- Retrieve Top Context Passages ---
def retrieve_context(question: str, top_k: int = 3):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    return "\n".join([texts[i] for i in I[0]])

# --- Main Query Function ---
def query_document(question: str) -> str:
    setup_agent()
    context = retrieve_context(question)
    prompt = f"Answer the question based on the following document content:\n\n{context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return response.content

# --- Entry Point: Upload + Process Document ---
def process_document(file_path: str):
    text = extract_text_from_pdf(file_path)
    embed_text_chunks(text)