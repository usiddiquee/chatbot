from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os
import uuid
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from groq import Groq  # Import Groq client
import os
from dotenv import load_dotenv
load_dotenv()

# Define sanitize_text function to handle encoding issues
def sanitize_text(text):
    """
    Sanitize text to ensure it can be properly encoded and decoded.
    
    Args:
        text (str): The text to sanitize
        
    Returns:
        str: Sanitized text
    """
    if text is None:
        return ""
    # Remove or replace problematic characters
    return text.encode('utf-8', errors='replace').decode('utf-8')

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="PDF Document Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("grok_api"))  # Ensure you set your API key in the environment
LLM_MODEL = "llama-3.3-70b-versatile"  # You can change this to other models as needed

# Create directories for storage
os.makedirs("uploads", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

class Query(BaseModel):
    text: str
    document_id: str
    top_k: int = 5

class Document(BaseModel):
    id: str
    filename: str

class Documents(BaseModel):
    documents: List[Document]

class Chunk(BaseModel):
    text: str
    page: int
    position: int

def sliding_window_chunking(text, window_size=200, stride=100):
    """
    Chunks text using a sliding window approach.
    
    Args:
        text (str): The text to chunk
        window_size (int): The number of words in each chunk
        stride (int): The number of words to slide the window by
        
    Returns:
        List[str]: A list of text chunks
    """
    # Tokenize text into words
    words = word_tokenize(text)
    
    # Initialize chunks
    chunks = []
    
    # Exit early if text is too short
    if len(words) < window_size:
        return [text]
    
    # Create chunks using sliding window
    for i in range(0, len(words) - window_size + 1, stride):
        chunk = ' '.join(words[i:i + window_size])
        chunks.append(chunk)
    
    # Add the final chunk if there are remaining words
    if i + stride < len(words):
        chunks.append(' '.join(words[i + stride:]))
    
    return chunks

# Serve the HTML frontend at the root path
@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse('index.html')

# Mount static files if you have any (CSS, JS, images, etc.)
# Uncomment the following lines if you have a static folder
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload/", response_model=Document)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file, extract text, create embeddings, and store them."""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique ID for document
    document_id = str(uuid.uuid4())
    
    # Save PDF file
    file_path = os.path.join("uploads", f"{document_id}.pdf")
    with open(file_path, "wb") as pdf_file:
        pdf_file.write(await file.read())
    
    try:
        # Process PDF and extract text with proper UTF-8 encoding handling
        full_text_by_page = {}
        pdf_document = fitz.open(file_path)
        
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            # Handle encoding issues by replacing problematic characters
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            full_text_by_page[page_num] = text
        
        # Create chunks using sliding window approach
        chunks = []
        position = 0
        
        for page_num, page_text in full_text_by_page.items():
            # Apply sliding window chunking
            page_text = sanitize_text(page_text)
            page_chunks = sliding_window_chunking(page_text, window_size=200, stride=100)
            
            for chunk_text in page_chunks:
                # Skip very short chunks
                if len(chunk_text) < 30:
                    continue
                
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "position": position
                })
                position += 1
        
        # Create embeddings for chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts)
        
        # Create data structure with text chunks and embeddings
        data = {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": chunks,
            "embeddings": [emb.tolist() for emb in embeddings]
        }
        
        # Save embeddings to JSON file
        with open(os.path.join("embeddings", f"{document_id}.json"), "w", encoding="utf-8", errors="replace") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return {"id": document_id, "filename": file.filename}
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(os.path.join("embeddings", f"{document_id}.json")):
            os.remove(os.path.join("embeddings", f"{document_id}.json"))
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/documents/", response_model=Documents)
def list_documents():
    """List all available documents."""
    documents = []
    
    for filename in os.listdir("embeddings"):
        if filename.endswith(".json"):
            document_id = filename[:-5]  # Remove .json extension
            
            # Read the document info from the embedding file
            try:
                with open(os.path.join("embeddings", filename), "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                    documents.append({
                        "id": document_id,
                        "filename": data.get("filename", "Unknown")
                    })
            except:
                # If we can't read the file, skip it
                continue
    
    return {"documents": documents}

@app.post("/query/", response_model=Dict[str, Any])
def query_document(query: Query):
    """Query a document with a text prompt using cosine similarity to find relevant chunks."""
    
    # Check if document exists
    embedding_path = os.path.join("embeddings", f"{query.document_id}.json")
    if not os.path.exists(embedding_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Load document data
        with open(embedding_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        
        # Create query embedding
        query_embedding = model.encode(query.text)
        
        # Load document embeddings
        doc_embeddings = np.array(data["embeddings"])
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-query.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": data["chunks"][idx],
                "similarity": float(similarities[idx])
            })
        
        return {
            "document_id": query.document_id,
            "filename": data["filename"],
            "query": query.text,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.post("/chatbot/", response_model=Dict[str, Any])
async def chatbot_query(query: Query):
    """
    Chatbot endpoint that uses retrieved context from PDF to answer questions
    and generates responses using the Groq LLM API (Llama-3.3-70b).
    """
    # First retrieve relevant context using our query endpoint
    retrieval_results = query_document(query)
    
    # Extract the top context chunks
    contexts = []
    for item in retrieval_results["results"]:
        try:
            # Ensure the chunk text is properly sanitized
            chunk_text = sanitize_text(item["chunk"]["text"])
            contexts.append(chunk_text)
        except Exception as e:
            # Skip problematic chunks but log them
            print(f"Error processing chunk: {e}")
    
    # Join contexts with proper formatting
    context_text = "\n\n".join([sanitize_text(ctx) for ctx in contexts])
    
    # Prepare the prompt for the LLM with enhanced persona and tag prompting
    prompt = f"""<instructions>
You are DocumentExpert, a professional document analysis assistant specialized in extracting accurate information from texts.
Your responses should be:
- Precise and factual, based solely on the provided context
- Well-structured with clear sections where appropriate
- Written in a helpful, professional tone
- Include direct quotes from the source material when relevant (using quotation marks)
- Free of speculation or information not contained in the provided context

Never make up information. If the context doesn't contain relevant information to answer the question,
explicitly state this limitation and explain what specific information would be needed to provide a proper answer.
</instructions>

<context>
{sanitize_text(context_text)}
</context>

<question>
{sanitize_text(query.text)}
</question>

<answer_format>
1. Start with a direct response to the question
2. Provide supporting evidence from the context
3. If needed, organize information into clear sections
4. Cite specific parts of the document when possible (page numbers, sections)
5. If the question cannot be fully answered from the context, clearly state what information is missing
</answer_format>

<answer>"""

    try:
        # Send the prompt to Groq LLM API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are DocumentExpert, a professional document analysis assistant that answers questions based exclusively on provided context. 
                    You excel at extracting precise information from documents and presenting it in a clear, structured format.
                    When citing information, you reference specific sections or pages and use direct quotes when appropriate.
                    You maintain professional integrity by refusing to speculate beyond what's explicitly stated in the document."""
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=LLM_MODEL
        )
        
        # Extract the LLM response
        answer = chat_completion.choices[0].message.content
        
        # If the answer contains the closing tag, extract only the content between tags
        if "</answer>" in answer:
            answer = answer.split("<answer>")[1].split("</answer>")[0].strip()
        
        response = {
            "document_id": query.document_id,
            "query": query.text,
            "answer": answer,
            "context": context_text,
            "sources": [{"page": item["chunk"]["page"], "position": item["chunk"]["position"]} 
                      for item in retrieval_results["results"]]
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from LLM: {str(e)}")

@app.get("/extract-chunks/{document_id}")
def get_document_chunks(document_id: str):
    """Get all chunks from a specific document."""
    embedding_path = os.path.join("embeddings", f"{document_id}.json")
    if not os.path.exists(embedding_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        with open(embedding_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        
        return {
            "document_id": document_id,
            "filename": data["filename"],
            "total_chunks": len(data["chunks"]),
            "chunks": data["chunks"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)