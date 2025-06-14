import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import base64
import hashlib
import numpy as np
from datetime import datetime
from config.settings import settings
import tempfile
import io

# Try to import enhanced components
try:
    from src.models.vector_store_complete import ComprehensiveVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="TDS Virtual TA",
    description="A Virtual Teaching Assistant for Tools in Data Science course",
    version="1.0.0"
)
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Vercel!"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
knowledge_base = {}
chunks = []
embeddings = []

@app.on_event("startup")
async def load_knowledge_base():
    """Load the scraped TDS course content and embeddings on startup"""
    global knowledge_base, chunks, embeddings
    
    try:
        data_path = os.path.join(settings.RAW_DATA_PATH, 'tds_course_all.json')
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            print(f"SUCCESS: Loaded knowledge base with {len(knowledge_base)} sections")
            
            total_chars = sum(len(section.get('content', '')) for section in knowledge_base.values())
            print(f"Total content: {total_chars:,} characters")
            
            for section_name, section_data in knowledge_base.items():
                content_length = len(section_data.get('content', ''))
                print(f"  {section_name}: {content_length:,} characters")
        else:
            print(f"ERROR: Knowledge base file not found: {data_path}")
            knowledge_base = {}
        
        # Load embeddings if available
        embeddings_file = 'data/processed/comprehensive_embeddings.npz'
        if os.path.exists(embeddings_file):
            data = np.load(embeddings_file, allow_pickle=True)
            chunks = data['content'].tolist()
            embeddings = data['embeddings']
            print(f"SUCCESS: Loaded {len(chunks)} chunks and embeddings")
        else:
            print("WARNING: No embeddings file found - using basic search")
        
    except Exception as e:
        print(f"ERROR: Error loading knowledge base: {e}")
        knowledge_base = {}

def get_embeddings(text):
    """Create embedding using Gemini with fallback"""
    try:
        if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
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
    
    # Fallback: create deterministic hash-based embedding
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    embedding = np.array([int(text_hash[i:i+2], 16) / 255.0 for i in range(0, min(768, len(text_hash)), 2)])
    
    if len(embedding) < 384:
        embedding = np.pad(embedding, (0, 384 - len(embedding)), 'constant')
    else:
        embedding = embedding[:384]
        
    return embedding

def get_image_description(image_data):
    """Get image description using Gemini"""
    try:
        if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            from PIL import Image
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            prompt = """Describe the content of this image in detail, focusing on any text, 
            objects, or technical content that would be relevant for answering questions 
            about Tools in Data Science course topics like Docker, Git, Python, or data science.
            Be very specific about any error messages, code snippets, or technical diagrams shown."""
            
            response = model.generate_content([prompt, image])
            return response.text
    except Exception as e:
        print(f"Image description failed: {e}")
    
    return "Image provided but could not be processed. The image likely contains relevant technical information for the TDS course question."

def generate_llm_response(question, context):
    """Generate response using Gemini"""
    try:
        if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""You are a helpful teaching assistant for the Tools in Data Science course.
            
            Question: {question}
            
            Context: {context}
            
            Provide a concise, helpful answer based on the context provided. Be specific and practical.
            Keep the response under 200 words."""
            
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        print(f"LLM response generation failed: {e}")
    
    return f"Based on the TDS course materials:\n\n{context[:500]}..."

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "TDS Virtual TA API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "knowledge_base_sections": len(knowledge_base),
        "chunks_loaded": len(chunks),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask")
async def ask_question(request: Request):
    """Main endpoint to ask questions to the Virtual TA"""
    try:
        data = await request.json()
        question = data.get("question", "")
        image = data.get("image")
        
        if not question.strip():
            return {"error": "Question cannot be empty"}
        
        # Process image if provided
        enhanced_question = question
        if image:
            print("Processing image...")
            image_description = get_image_description(image)
            enhanced_question = f"{question}\n\nImage context: {image_description}"
            print(f"Enhanced question with image context: {len(enhanced_question)} characters")
        
        # Get answer using vector search or basic search
        if len(chunks) > 0 and len(embeddings) > 0:
            answer_data = search_with_vector_store(enhanced_question)
        else:
            answer_data = search_knowledge_base_basic(enhanced_question)
        
        return {
            "answer": answer_data["answer"],
            "links": answer_data["links"]
        }
        
    except Exception as e:
        return {"error": f"Error processing question: {str(e)}"}

def search_with_vector_store(question):
    """Search using vector similarity"""
    try:
        question_embedding = get_embeddings(question)
        similarities = np.dot(embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-10:][::-1]
        top_chunks = [chunks[i] for i in top_indices[:3]]
        
        combined_content = "\n\n".join(top_chunks)
        response_text = generate_llm_response(question, combined_content)
        
        links = [
            {"url": "https://tds.s-anand.net/#/2025-01/", "text": "TDS Course Content"},
            {"url": "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34", "text": "TDS Discourse"}
        ]
        
        return {
            "answer": response_text,
            "links": links
        }
        
    except Exception as e:
        print(f"Vector search failed: {e}")
        return search_knowledge_base_basic(question)

def search_knowledge_base_basic(question):
    """Basic search through knowledge base"""
    question_lower = question.lower()
    relevant_sections = []
    
    for section_name, section_data in knowledge_base.items():
        content = section_data.get('content', '').lower()
        
        score = 0
        question_words = question_lower.split()
        
        for word in question_words:
            if len(word) > 2:
                if word in ['docker', 'git', 'python', 'container', 'repository', 'commit', 'gpt', 'model']:
                    score += content.count(word) * 3
                else:
                    score += content.count(word)
        
        if score > 0:
            relevant_sections.append({
                "section": section_name,
                "score": score,
                "content": section_data.get('content', ''),
                "url": section_data.get('url', '')
            })
    
    relevant_sections.sort(key=lambda x: x['score'], reverse=True)
    
    if not relevant_sections:
        return {
            "answer": "I couldn't find specific information about your question in the TDS course materials.",
            "links": []
        }
    
    best_match = relevant_sections[0]
    content = best_match['content']
    snippet = content[:800] + "..." if len(content) > 800 else content
    
    links = []
    for section in relevant_sections[:3]:
        section_name = section['section']
        
        if section_name == 'README':
            url = "https://tds.s-anand.net/#/2025-01/"
        elif section_name == 'docker':
            url = "https://tds.s-anand.net/#/2025-01/docker"
        elif section_name == 'git':
            url = "https://tds.s-anand.net/#/2025-01/git"
        else:
            url = f"https://tds.s-anand.net/#/2025-01/{section_name}"
            
        links.append({
            "url": url,
            "text": f"Information from {section_name} section of TDS course"
        })
    
    return {
        "answer": f"Based on the TDS course materials:\n\n{snippet}",
        "links": links[:3]
    }

if __name__ == "__main__":
    print("Starting TDS Virtual TA API with Gemini Integration...")
    print(f"Server will run on http://{settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
