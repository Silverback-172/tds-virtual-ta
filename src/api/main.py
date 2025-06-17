import sys
import os
import json
import base64
import hashlib
import io
import numpy as np
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from config.settings import settings

# Allow importing from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Optional import for enhanced vector search
try:
    from src.models.vector_store_complete import ComprehensiveVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(
    title="TDS Virtual TA",
    description="A Virtual Teaching Assistant for Tools in Data Science course",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None
    context: Optional[str] = "general"

class TAResponse(BaseModel):
    answer: str
    links: List[Dict[str, str]]

# Global state
knowledge_base = {}
chunks = []
embeddings = []

@app.on_event("startup")
async def load_knowledge_base():
    global knowledge_base, chunks, embeddings
    try:
        # Load course content
        data_path = os.path.join("data", "raw", 'tds_course_all.json')
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            print(f"Loaded {len(knowledge_base)} sections from knowledge base")

            total_chars = sum(len(s.get('content', '')) for s in knowledge_base.values())
            print(f"Total content: {total_chars} characters")

            for name, data in knowledge_base.items():
                print(f"{name}: {len(data.get('content', ''))} characters")
        else:
            print("Knowledge base file not found.")
            knowledge_base = {}

        # Load vector embeddings
        embeddings_path = 'data/processed/comprehensive_embeddings.npz'
        if os.path.exists(embeddings_path):
            data = np.load(embeddings_path, allow_pickle=True)
            chunks = data['content'].tolist()
            embeddings = data['embeddings']
            print(f"Loaded {len(chunks)} chunks and embeddings")
        else:
            print("No embeddings file found.")
    except Exception as e:
        print(f"Failed to load knowledge base: {e}")
        knowledge_base = {}

def get_embeddings(text):
    """Generate embeddings for text using Gemini or fallback to hash-based embedding"""
    try:
        if getattr(settings, 'GEMINI_API_KEY', '') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
    except Exception as e:
        print(f"Gemini embedding failed: {e}")
    
    # Fallback: deterministic hash embedding
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    embedding = np.array([int(text_hash[i:i+2], 16) / 255.0 for i in range(0, min(768, len(text_hash)), 2)])
    return np.pad(embedding, (0, 384 - len(embedding)), 'constant')[:384]

def get_image_description(image_data):
    """Process image using Gemini 2.0 Flash model"""
    try:
        if getattr(settings, 'GEMINI_API_KEY', '') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Use Gemini 2.0 Flash for image processing
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = """
            Analyze this image in the context of Tools in Data Science (TDS) course.
            Describe what you see and how it relates to data science concepts, programming, 
            or course materials. Be specific and educational.
            """
            
            response = model.generate_content([prompt, image])
            return response.text
    except Exception as e:
        print(f"Image processing failed: {e}")
        return "Image processing failed. Please describe your question in text."

def search_knowledge_base(query, top_k=5):
    """Search knowledge base using vector similarity"""
    try:
        if len(embeddings) == 0:
            return []
        
        # Get query embedding
        query_embedding = get_embeddings(query)
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((similarity, i))
        
        # Sort by similarity and get top results
        similarities.sort(reverse=True)
        top_results = similarities[:top_k]
        
        results = []
        for similarity, idx in top_results:
            if idx < len(chunks):
                results.append({
                    'content': chunks[idx],
                    'similarity': float(similarity),
                    'index': idx
                })
        
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []

def generate_response(question, context_results, image_description=None):
    """Generate response using Gemini or fallback to template"""
    try:
        if getattr(settings, 'GEMINI_API_KEY', '') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Prepare context
            context_text = "\n\n".join([result['content'] for result in context_results[:3]])
            
            prompt = f"""
            You are a Virtual Teaching Assistant for the Tools in Data Science (TDS) course.
            
            Question: {question}
            
            {f"Image Description: {image_description}" if image_description else ""}
            
            Context from course materials:
            {context_text}
            
            Please provide a helpful, accurate answer based on the course materials. 
            Be specific and educational. If the question is about course logistics, 
            assignments, or technical concepts, provide detailed guidance.
            """
            
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        print(f"Response generation failed: {e}")
    
    # Fallback response
    if context_results:
        return f"Based on the course materials, here's what I found relevant to your question: {context_results[0]['content'][:500]}..."
    else:
        return "I understand your question about the TDS course. While I don't have specific information readily available, I recommend checking the course materials or asking on the discourse forum for detailed guidance."

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "TDS Virtual TA API is running!",
        "status": "healthy",
        "knowledge_base_loaded": len(knowledge_base) > 0,
        "embeddings_loaded": len(embeddings) > 0
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest) -> TAResponse:
    """Main endpoint for asking questions to the Virtual TA"""
    try:
        question = request.question
        image_data = request.image
        context = request.context
        
        # Process image if provided
        image_description = None
        if image_data:
            image_description = get_image_description(image_data)
            # Combine question with image description for better search
            search_query = f"{question} {image_description}"
        else:
            search_query = question
        
        # Search knowledge base
        context_results = search_knowledge_base(search_query, top_k=5)
        
        # Generate response
        answer = generate_response(question, context_results, image_description)
        
        # Prepare links
        links = [
            {
                "url": "https://tds.s-anand.net/#/2025-01/",
                "text": "TDS Course Content"
            },
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34",
                "text": "TDS Discourse"
            }
        ]
        
        return TAResponse(answer=answer, links=links)
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "knowledge_base_sections": len(knowledge_base),
        "total_chunks": len(chunks),
        "embeddings_shape": embeddings.shape if len(embeddings) > 0 else "No embeddings",
        "vector_store_available": VECTOR_STORE_AVAILABLE,
        "gemini_configured": bool(getattr(settings, 'GEMINI_API_KEY', '') and settings.GEMINI_API_KEY != "your_gemini_api_key_here")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=getattr(settings, 'API_HOST', '0.0.0.0'), 
        port=getattr(settings, 'API_PORT', 8000)
    )
