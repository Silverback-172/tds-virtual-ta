import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import json
import requests
import glob
from datetime import datetime
from config.settings import settings
from bs4 import BeautifulSoup
import hashlib

class ComprehensiveVectorStore:
    def __init__(self):
        self.embeddings_file = 'data/processed/comprehensive_embeddings.npz'
        
    def create_embedding(self, text):
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
    
    def chunk_content(self, content, chunk_size=500, overlap=50):
        """Split content into overlapping chunks for better retrieval"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def clean_html_content(self, html_content):
        """Clean HTML content to extract readable text"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
            
            return cleaned_text
        except Exception as e:
            print(f"WARNING: HTML cleaning error: {e}")
            return html_content
    
    def create_comprehensive_embeddings(self):
        """Create embeddings from all sources using NumPy method"""
        print("Creating comprehensive embeddings using NumPy archive method...")
        
        all_content = []
        all_embeddings = []
        all_metadata = []
        
        # Process TDS course content
        print("Processing TDS course content...")
        data_path = os.path.join(settings.RAW_DATA_PATH, 'tds_course_all.json')
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            for section_name, section_data in knowledge_base.items():
                content = section_data.get('content', '')
                
                if content:
                    chunks = self.chunk_content(content)
                    
                    for i, chunk in enumerate(chunks):
                        embedding = self.create_embedding(chunk)
                        
                        all_content.append(chunk)
                        all_embeddings.append(embedding)
                        all_metadata.append({
                            'source': section_name,
                            'chunk_id': i,
                            'url': section_data.get('url', ''),
                            'type': 'course_content',
                            'scraped_at': section_data.get('scraped_at', ''),
                            'section': section_name
                        })
                        
                        print(f"  SUCCESS: Processed chunk {i+1}/{len(chunks)} from {section_name}")
        
        # Process Discourse content
        print("Processing Discourse content...")
        discourse_files = glob.glob(os.path.join(settings.RAW_DATA_PATH, 'discourse_topic_*.json'))
        
        discourse_chunks_count = 0
        for file_path in discourse_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    topic_data = json.load(f)
                
                topic_id = topic_data.get('id')
                topic_title = topic_data.get('title', '')
                topic_url = topic_data.get('url', '')
                
                posts = topic_data.get('posts', [])
                for post in posts:
                    raw_content = post.get('raw_content', '')
                    cleaned_content = post.get('cleaned_text', '')
                    
                    content = raw_content if raw_content else cleaned_content
                    
                    if content and len(content) > 50:
                        chunks = self.chunk_content(content)
                        
                        for i, chunk in enumerate(chunks):
                            embedding = self.create_embedding(chunk)
                            
                            all_content.append(chunk)
                            all_embeddings.append(embedding)
                            all_metadata.append({
                                'source': f"discourse_topic_{topic_id}",
                                'chunk_id': i,
                                'url': topic_url,
                                'type': 'discourse_post',
                                'topic_id': topic_id,
                                'topic_title': topic_title,
                                'post_number': post.get('post_number', 1),
                                'username': post.get('username', 'unknown'),
                                'created_at': post.get('created_at', ''),
                                'section': 'discourse'
                            })
                            
                            discourse_chunks_count += 1
                
                print(f"  SUCCESS: Processed topic {topic_id}: {topic_title}")
                
            except Exception as e:
                print(f"WARNING: Error processing {file_path}: {e}")
        
        print(f"Processed {discourse_chunks_count} discourse chunks from {len(discourse_files)} topics")
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        
        # Save as NumPy archive (TA's recommended method)
        print("Saving comprehensive embeddings using NumPy archive...")
        np.savez_compressed(
            self.embeddings_file,
            embeddings=np.array(all_embeddings),
            content=np.array(all_content, dtype=object),
            metadata=np.array(all_metadata, dtype=object)
        )
        
        file_size = os.path.getsize(self.embeddings_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print("SUCCESS: Comprehensive embeddings saved successfully!")
        print(f"Total chunks: {len(all_embeddings)}")
        print(f"Course content chunks: {len([m for m in all_metadata if m['type'] == 'course_content'])}")
        print(f"Discourse chunks: {len([m for m in all_metadata if m['type'] == 'discourse_post'])}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Embedding dimensions: {len(all_embeddings[0]) if all_embeddings else 0}")
        
        if file_size_mb > 15:
            print("WARNING: File size exceeds 15MB recommendation")
        else:
            print("SUCCESS: File size within TA's 15MB recommendation")
        
        return {
            'total_chunks': len(all_embeddings),
            'course_chunks': len([m for m in all_metadata if m['type'] == 'course_content']),
            'discourse_chunks': len([m for m in all_metadata if m['type'] == 'discourse_post']),
            'file_size_mb': file_size_mb,
            'embedding_dimensions': len(all_embeddings[0]) if all_embeddings else 0
        }
    
    def load_embeddings(self):
        """Load embeddings from NumPy archive"""
        if not os.path.exists(self.embeddings_file):
            print("ERROR: Comprehensive embeddings file not found. Run create_comprehensive_embeddings() first.")
            return None
        
        try:
            data = np.load(self.embeddings_file, allow_pickle=True)
            print(f"SUCCESS: Loaded comprehensive embeddings: {len(data['embeddings'])} chunks")
            return {
                'embeddings': data['embeddings'],
                'content': data['content'],
                'metadata': data['metadata']
            }
        except Exception as e:
            print(f"ERROR: Error loading comprehensive embeddings: {e}")
            return None
    
    def search_similar(self, query, top_k=10, filter_type=None):
        """Search for similar content using cosine similarity"""
        embeddings_data = self.load_embeddings()
        if not embeddings_data:
            return []
        
        query_embedding = self.create_embedding(query)
        similarities = np.dot(embeddings_data['embeddings'], query_embedding)
        
        valid_indices = range(len(similarities))
        if filter_type:
            valid_indices = [
                i for i, metadata in enumerate(embeddings_data['metadata'])
                if metadata['type'] == filter_type
            ]
        
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in valid_similarities[:top_k]]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': embeddings_data['content'][idx],
                'metadata': embeddings_data['metadata'][idx],
                'similarity': float(similarities[idx])
            })
        
        return results

if __name__ == "__main__":
    vector_store = ComprehensiveVectorStore()
    
    print("Starting comprehensive embeddings creation...")
    stats = vector_store.create_comprehensive_embeddings()
    
    print("\nTesting search functionality...")
    results = vector_store.search_similar("How do I use Docker in TDS?", top_k=5)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Type: {result['metadata']['type']}")
        print(f"Content: {result['content'][:200]}...")
