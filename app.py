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
import logging
import traceback
import sys
from datetime import datetime

load_dotenv()

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Silence verbose libraries
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

def sanitize_text(text):
    """Sanitize text to ensure it can be properly encoded and decoded."""
    if text is None:
        return ""
    
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.error(f"Error sanitizing text: {e}")
        return ""

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK tokenizers...")
        try:
            nltk.download('punkt_tab')
            nltk.download('punkt')
        except Exception as e:
            logger.error(f"Failed to download NLTK tokenizers: {e}")
            raise

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
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    raise

# Initialize Groq client
try:
    groq_api_key = os.getenv("grok_api")
    if not groq_api_key:
        raise ValueError("GROQ API key not found in environment variables")
    
    groq_client = Groq(api_key=groq_api_key)
    LLM_MODEL = "llama-3.3-70b-versatile"
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise

# Create directories for storage
try:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create storage directories: {e}")
    raise

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
    """Chunks text using a sliding window approach."""
    try:
        words = word_tokenize(text)
        chunks = []
        
        if len(words) < window_size:
            return [text]
        
        for i in range(0, len(words) - window_size + 1, stride):
            chunk = ' '.join(words[i:i + window_size])
            chunks.append(chunk)
        
        # Add the final chunk if there are remaining words
        if i + stride < len(words):
            final_chunk = ' '.join(words[i + stride:])
            chunks.append(final_chunk)
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error in sliding window chunking: {e}")
        raise

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    try:
        if os.path.exists('index.html'):
            return FileResponse('index.html')
        else:
            raise HTTPException(status_code=404, detail="index.html not found")
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        raise

@app.post("/upload/", response_model=Document)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file, extract text, create embeddings, and store them."""
    
    logger.info(f"Processing PDF upload: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    document_id = str(uuid.uuid4())
    file_path = os.path.join("uploads", f"{document_id}.pdf")
    
    try:
        # Save uploaded file
        file_content = await file.read()
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(file_content)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        # Process PDF and extract text
        full_text_by_page = {}
        
        pdf_document = fitz.open(file_path)
        logger.info(f"Processing PDF with {pdf_document.page_count} pages")
        
        for page_num, page in enumerate(pdf_document):
            try:
                text = page.get_text()
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                full_text_by_page[page_num] = text
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                full_text_by_page[page_num] = ""
        
        pdf_document.close()
        
        # Create chunks using sliding window approach
        chunks = []
        position = 0
        
        for page_num, page_text in full_text_by_page.items():
            try:
                page_text = sanitize_text(page_text)
                if not page_text.strip():
                    continue
                    
                page_chunks = sliding_window_chunking(page_text, window_size=200, stride=100)
                
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    # Skip very short chunks
                    if len(chunk_text) < 30:
                        continue
                    
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "position": position
                    })
                    position += 1
                    
            except Exception as e:
                logger.error(f"Error chunking page {page_num + 1}: {e}")
                continue
        
        logger.info(f"Created {len(chunks)} text chunks")
        
        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")
        
        # Create embeddings for chunks
        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = model.encode(texts)
            logger.info(f"Generated embeddings for {len(texts)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
        
        # Save embeddings to JSON file
        data = {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": chunks,
            "embeddings": [emb.tolist() for emb in embeddings]
        }
        
        embeddings_file = os.path.join("embeddings", f"{document_id}.json")
        
        try:
            with open(embeddings_file, "w", encoding="utf-8", errors="replace") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            if not os.path.exists(embeddings_file):
                raise Exception("Failed to save embeddings file")
                
        except Exception as e:
            logger.error(f"Error saving embeddings file: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving embeddings: {str(e)}")
        
        logger.info(f"Successfully processed document: {document_id}")
        return {"id": document_id, "filename": file.filename}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        
        # Clean up on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            embeddings_file = os.path.join("embeddings", f"{document_id}.json")
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up files: {cleanup_error}")
            
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/documents/", response_model=Documents)
def list_documents():
    """List all available documents."""
    documents = []
    
    try:
        embeddings_dir = "embeddings"
        if not os.path.exists(embeddings_dir):
            return {"documents": documents}
        
        files = os.listdir(embeddings_dir)
        
        for filename in files:
            if filename.endswith(".json"):
                document_id = filename[:-5]  # Remove .json extension
                
                try:
                    filepath = os.path.join(embeddings_dir, filename)
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        data = json.load(f)
                        documents.append({
                            "id": document_id,
                            "filename": data.get("filename", "Unknown")
                        })
                except Exception as e:
                    logger.error(f"Error reading document file {filename}: {e}")
                    continue
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/query/", response_model=Dict[str, Any])
def query_document(query: Query):
    """Query a document with a text prompt using cosine similarity to find relevant chunks."""
    
    embedding_path = os.path.join("embeddings", f"{query.document_id}.json")
    if not os.path.exists(embedding_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Load document data
        with open(embedding_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        
        # Create query embedding
        query_embedding = model.encode(query.text)
        
        # Load document embeddings and calculate similarity
        doc_embeddings = np.array(data["embeddings"])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-query.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            results.append({
                "chunk": data["chunks"][idx],
                "similarity": similarity_score
            })
        
        return {
            "document_id": query.document_id,
            "filename": data["filename"],
            "query": query.text,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error querying document: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.post("/chatbot/", response_model=Dict[str, Any])
async def chatbot_query(query: Query):
    """
    Chatbot endpoint that uses retrieved context from PDF to answer questions
    and generates responses using the Groq LLM API (Llama-3.3-70b).
    """
    logger.info(f"Processing chatbot query for document {query.document_id}")
    
    try:
        # First retrieve relevant context
        retrieval_results = query_document(query)
        
        # Extract context chunks
        contexts = []
        for item in retrieval_results["results"]:
            try:
                chunk_text = sanitize_text(item["chunk"]["text"])
                if chunk_text.strip():
                    contexts.append(chunk_text)
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")
        
        context_text = "\n\n".join([sanitize_text(ctx) for ctx in contexts])
        
        # Prepare the prompt for the LLM
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
        
        # Send request to Groq LLM API
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
        logger.error(f"Error generating response from LLM: {e}")
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
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "uploads_exists": os.path.exists("uploads"),
            "embeddings_exists": os.path.exists("embeddings"),
            "uploads_writable": os.access("uploads", os.W_OK) if os.path.exists("uploads") else False,
            "embeddings_writable": os.access("embeddings", os.W_OK) if os.path.exists("embeddings") else False
        }
    }

if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)