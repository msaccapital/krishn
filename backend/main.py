from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import tempfile
import os
import shutil
import uuid
from PyPDF2 import PdfReader
import time
import numpy as np

# Create FastAPI app
app = FastAPI(title="Multi-PDF Krishn API", version="1.0")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class UploadResponse(BaseModel):
    session_id: str
    file_name: str
    pages_processed: int
    status: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    response_time: float
    confidence: float

# Main AI System Class
class KrishnMultiPDF:
    def __init__(self):
        self.sessions = {}
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.is_models_loaded = False
    
    def load_models(self):
        """Load AI models once at startup"""
        if self.is_models_loaded:
            return True
            
        print("🚀 Loading Mistral 7B...")
        try:
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with 4-bit quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.is_models_loaded = True
            print("✅ AI Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def process_pdf(self, file_path: str, session_id: str):
        """Process uploaded PDF and create vector database"""
        try:
            print(f"📄 Processing PDF: {file_path}")
            reader = PdfReader(file_path)
            chunks = []
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Clean and chunk the text
                    cleaned_text = ' '.join(text.split())
                    chunks.append({
                        'text': cleaned_text[:1000],  # Limit text length
                        'source': os.path.basename(file_path),
                        'page_number': page_num + 1
                    })
                    print(f"   Processed page {page_num + 1}")
            
            # Create vector database
            if chunks:
                texts = [chunk['text'] for chunk in chunks]
                print(f"🔍 Creating embeddings for {len(texts)} chunks...")
                embeddings = self.embedder.encode(texts)
                
                # Convert to numpy array if needed
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                
                # Store in session
                self.sessions[session_id] = {
                    'index': index,
                    'chunks': chunks,
                    'files': [os.path.basename(file_path)],
                    'embeddings': embeddings
                }
                
                print(f"✅ PDF processing complete: {len(chunks)} chunks indexed")
                return len(chunks)
            else:
                print("❌ No text found in PDF")
                return 0
                
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return 0
    
    def search_documents(self, session_id: str, query: str, top_k: int = 3):
        """Search in session's vector database"""
        if session_id not in self.sessions:
            return []
        
        session_data = self.sessions[session_id]
        query_embedding = self.embedder.encode([query])
        
        # Ensure query_embedding is numpy array and normalized
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = session_data['index'].search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(session_data['chunks']):
                chunk = session_data['chunks'][idx]
                results.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'page_number': chunk['page_number'],
                    'score': float(score)
                })
        
        return results
    
    def generate_answer(self, session_id: str, question: str, max_length: int = 300):
        """Generate answer using relevant PDF content"""
        start_time = time.time()
        
        if not self.is_models_loaded:
            return {
                "answer": "AI models are still loading. Please try again in a moment.",
                "sources": [],
                "response_time": time.time() - start_time,
                "confidence": 0.0
            }
        
        # Search for relevant content
        relevant_chunks = self.search_documents(session_id, question, top_k=3)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in your uploaded documents.",
                "sources": [],
                "response_time": time.time() - start_time,
                "confidence": 0.0
            }
        
        # Build context from found chunks
        context = "\n\n".join([
            f"From {chunk['source']} page {chunk['page_number']}: {chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        prompt = f"""<s>[INST] Using ONLY the information below, answer the question. 
If the answer cannot be found in the information, say "I couldn't find relevant information."

INFORMATION:
{context}

QUESTION: {question}

ANSWER: [/INST]"""
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            answer = response[0]['generated_text'].strip()
            
            # Calculate confidence
            avg_confidence = sum([chunk['score'] for chunk in relevant_chunks]) / len(relevant_chunks)
            
            return {
                "answer": answer,
                "sources": relevant_chunks,
                "response_time": time.time() - start_time,
                "confidence": avg_confidence
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "response_time": time.time() - start_time,
                "confidence": 0.0
            }

# Initialize the multi-PDF system
krishn_system = KrishnMultiPDF()

@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    print("🔄 Starting up Krishn AI...")
    krishn_system.load_models()

@app.get("/")
async def root():
    return {"message": "🚀 Multi-PDF Krishn API is running!"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload and processing"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process PDF
        pages_processed = krishn_system.process_pdf(temp_path, session_id)
        
        # Cleanup
        os.unlink(temp_path)
        
        if pages_processed == 0:
            raise HTTPException(status_code=400, detail="Could not process PDF - no text found or PDF is corrupted")
        
        return UploadResponse(
            session_id=session_id,
            file_name=file.filename,
            pages_processed=pages_processed,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """Chat with the uploaded PDF"""
    if not request.session_id or request.session_id not in krishn_system.sessions:
        raise HTTPException(status_code=400, detail="Invalid session ID - please upload a PDF first")
    
    try:
        result = krishn_system.generate_answer(request.session_id, request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session-info/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session"""
    if session_id not in krishn_system.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = krishn_system.sessions[session_id]
    return {
        "files": session['files'],
        "chunks_count": len(session['chunks']),
        "active": True
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "sessions_active": len(krishn_system.sessions),
        "models_loaded": krishn_system.is_models_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
