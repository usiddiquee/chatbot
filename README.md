# PDF Document Assistant
A powerful AI-powered document assistant that allows you to upload PDF files, extract content, and interact with your documents through an intelligent chatbot interface. Built with FastAPI backend and a responsive web frontend.

# Features
PDF Upload: Upload and process PDF documents with automatic text extraction
Intelligent Chunking: Advanced sliding window text chunking for optimal context retrieval
Semantic Search: Vector-based similarity search using sentence transformers
AI Chatbot: Ask questions about your documents and get intelligent answers using Groq's Llama model
Document Management: View and manage uploaded documents
Responsive UI: Clean, modern web interface built with Bootstrap

# Architecture

Backend: FastAPI with sentence transformers for embeddings
AI Model: Integration with Groq's Llama-3.3-70b-versatile
Text Processing: PyMuPDF for PDF extraction, NLTK for tokenization
Storage: Local file system with JSON-based embeddings
Frontend: HTML/CSS/JavaScript with Bootstrap and jQuery

# Prerequisites

Python 3.8+
A Groq API key (get one at https://console.groq.com)

# Installation

# Clone the repository
bashgit clone https://github.com/your-username/pdf-document-assistant.git
cd pdf-document-assistant

# Create a virtual environment
bashpython -m venv pdf_assistant_env
source pdf_assistant_env/bin/activate  # On Windows: pdf_assistant_env\Scripts\activate

# Install dependencies
bashpip install fastapi uvicorn sentence-transformers numpy scikit-learn
pip install PyMuPDF nltk groq python-dotenv

# Set up environment variables
Create a .env file in the project root:
envgrok_api=your_groq_api_key_here

# Download NLTK data
The application will automatically download required NLTK data on first run.

# Usage
# Starting the Backend Server

# Run the FastAPI server
bashpython app.py
The server will start at http://localhost:8000
# Verify installation
Visit http://localhost:8000 to see the API welcome message

# Using the Web Interface

# Open the frontend
Open index.html in your web browser, or serve it using a simple HTTP server:
bashpython -m http.server 3000
Then visit http://localhost:3000
# Upload a PDF

Click "Choose File" and select a PDF document
Click "Upload" to process the document
Wait for the upload to complete


# Chat with your document

Select an uploaded document from the list
Type your question in the chat input
Press Enter or click the send button
Get AI-powered answers based on your document content



# API Endpoints
# Document Management

POST /upload/ - Upload and process a PDF file
GET /documents/ - List all uploaded documents
GET /extract-chunks/{document_id} - Get all text chunks from a document

# Querying

POST /query/ - Search document using semantic similarity
POST /chatbot/ - Ask questions and get AI-generated answers

Example API Usage
pythonimport requests

# Upload a PDF
with open('document.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/upload/', files={'file': f})
    document_id = response.json()['id']

# Ask a question
question_data = {
    "text": "What is the main topic of this document?",
    "document_id": document_id,
    "top_k": 5
}
response = requests.post('http://localhost:8000/chatbot/', json=question_data)
answer = response.json()['answer']

# Project Structure
pdf-document-assistant/
├── app.py                 # FastAPI backend application
├── index.html            # Web frontend interface
├── .env                  # Environment variables (not in repo)
├── requirements.txt      # Python dependencies
├── uploads/              # Uploaded PDF files (created automatically)
├── embeddings/           # Document embeddings storage (created automatically)
└── README.md            # This file

![image](https://github.com/user-attachments/assets/d18bef5f-5ba5-4155-aeff-96d36315de01)


# Configuration
Environment Variables

grok_api: Your Groq API key for LLM functionality

Customizable Parameters

LLM_MODEL: Change the Groq model (default: "llama-3.3-70b-versatile")
Chunking parameters: window_size (default: 200), stride (default: 100)
Similarity search: top_k parameter for number of results

# Troubleshooting

# Common Issues

"Document not found" error

Ensure the document was uploaded successfully
Check that the document ID is correct

# Upload fails

Verify the file is a valid PDF
Check file size limits
Ensure sufficient disk space


# Groq API errors

Verify your API key is correct
Check your Groq account limits
Ensure internet connectivity


# Frontend can't connect to backend

Verify the backend is running on port 8000
Check CORS settings if accessing from different domain
Update API_BASE_URL in index.html if needed



Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

FastAPI for the excellent web framework
Sentence Transformers for semantic embeddings
Groq for high-performance LLM inference
PyMuPDF for PDF processing
Bootstrap for the responsive UI

Support
If you encounter any issues or have questions, please:

Check the troubleshooting section above
Search existing GitHub issues
Create a new issue with detailed information about your problem

