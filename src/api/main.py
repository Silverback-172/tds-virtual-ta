import sys
import os
import json
import base64
import hashlib
import io
import numpy as np
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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

# Global state
knowledge_base = {}
chunks = []
embeddings = []

@app.on_event("startup")
async def load_knowledge_base():
    global knowledge_base, chunks, embeddings
    try:
        data_path = os.path.join(settings.RAW_DATA_PATH, 'tds_course_all.json')
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
    try:
        if getattr(settings, 'GEMINI_API_KEY', '') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            prompt = (
                "Describe the content of this image in detail, focusing on any text, "
                "objects, or technical content relevant to Tools in Data Science topics "
                "like Docker, Git, Python, or data science. Be specific about error messages, "
                "code snippets, or diagrams."
            )
            response = model.generate_content([prompt, image])
            return response.text
    except Exception as e:
        print(f"Image processing failed: {e}")
    
    return "Image was provided but could not be processed. It might contain relevant technical information."

def generate_llm_response(question, context):
    try:
        if getattr(settings, 'GEMINI_API_KEY', '') and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = (
                f"You are a helpful TA for the Tools in Data Science course.\n\n"
                f"Question: {question}\n\n"
                f"Context: {context}\n\n"
                f"Answer concisely and helpfully. Keep the response under 200 words."
            )
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        print(f"LLM generation failed: {e}")
    
    return f"Based on the course materials:\n\n{context[:500]}..."

@app.get("/")
async def root():
    return {
        "message": "TDS Virtual TA is running",
        "version": "1.0.0",
        "status": "healthy",
        "sections": len(knowledge_base),
        "chunks": len(chunks),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        image = data.get("image")

        if not question.strip():
            return {"error": "Question cannot be empty."}

        enhanced_question = question
        if image:
            description = get_image_description(image)
            enhanced_question += f"\n\nImage context: {description}"

        if chunks and embeddings:
            result = search_with_vector_store(enhanced_question)
        else:
            result = search_knowledge_base_basic(enhanced_question)

        return {
            "answer": result["answer"],
            "links": result["links"]
        }
    except Exception as e:
        return {"error": str(e)}

def search_with_vector_store(question):
    try:
        question_embedding = get_embeddings(question)
        similarities = np.dot(embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-10:][::-1]
        top_chunks = [chunks[i] for i in top_indices[:3]]

        context = "\n\n".join(top_chunks)
        answer = generate_llm_response(question, context)

        return {
            "answer": answer,
            "links": [
                {"url": "https://tds.s-anand.net/#/2025-01/", "text": "TDS Course Content"},
                {"url": "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34", "text": "TDS Discourse"}
            ]
        }
    except Exception as e:
        print(f"Vector search failed: {e}")
        return search_knowledge_base_basic(question)

def search_knowledge_base_basic(question):
    question_lower = question.lower()
    relevant = []

    for name, data in knowledge_base.items():
        content = data.get('content', '').lower()
        score = sum(content.count(word) * (3 if word in ['docker', 'git', 'python'] else 1)
                    for word in question_lower.split() if len(word) > 2)
        if score > 0:
            relevant.append({
                "section": name,
                "score": score,
                "content": data.get('content', ''),
                "url": data.get('url', '')
            })

    relevant.sort(key=lambda x: x['score'], reverse=True)
    if not relevant:
        return {
            "answer": "No relevant information found in the TDS course materials.",
            "links": []
        }

    top = relevant[0]
    snippet = top['content'][:800] + "..." if len(top['content']) > 800 else top['content']

    links = []
    for sec in relevant[:3]:
        sec_name = sec['section']
        links.append({
            "url": f"https://tds.s-anand.net/#/2025-01/{sec_name if sec_name != 'README' else ''}",
            "text": f"Info from {sec_name} section"
        })

    return {
        "answer": f"Based on the course:\n\n{snippet}",
        "links": links
    }

# Note: No __main__ block — Vercel uses its own ASGI handler
