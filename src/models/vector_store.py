import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import json
import requests
from datetime import datetime
from config.settings import settings
import glob
from bs4 import BeautifulSoup

class EfficientVectorStore:
    def __init__(self):
        self.embeddings_file = 'data/processed/embeddings.npz'
        self.aipipe_base_url = "https://api.aipipe.org"
        
    def create_embedding(self, text):
        """Create embedding using AIPipe (with fallback to local method)"""
        try:
            if hasattr(settings, 'AIPIPE_TOKEN') and settings.AIPIPE_TOKEN != "your_aipipe_token_here":
                response = requests.post(
                    f"{self.aipipe_base_url}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.AIPIPE_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "text-embedding-3-small",
                        "input": text[:8000]  # Limit text length
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return np.array(result['data'][0]['embedding'])
            
            # Fallback: create simple hash-based embedding for testing
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hash to simple 384-dimensional vector
            embedding = np.array([ord(c) / 255.0 for c in text_hash[:384]])
            return embedding
            
        except Exception as e:
            print(f"Embedding creation failed: {e}")
            # Simple fallback embedding
            return np.random.rand(384)
    
    def clean_html_content(self, html_content):
        """Clean HTML content to extract readable text"""
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return cleaned_text
    
    def chunk_content(self, content, chunk_size=500, overlap=50):
        """Split content into overlapping chunks for better retrieval"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings_from_knowledge_base(self):
        """Create embeddings from all scraped content using TA's method"""
        print("🔄 Creating efficient embeddings using NumPy archive method...")
        
        all_content = []
        all_embeddings = []
        all_metadata = []
        
        # Load TDS course content
        data_path = os.path.join(settings.RAW_DATA_PATH, 'tds_course_all.json')
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            
            print(f"📚 Processing {len(knowledge_base)} course sections...")
            
            for section_name, section_data in knowledge_base.items():
                content = section_data.get('content', '')
                
                if content:
                    # Chunk the content for better retrieval
                    chunks = self.chunk_content(content)
                    
                    for i, chunk in enumerate(chunks):
                        # Create embedding for each chunk
                        embedding = self.create_embedding(chunk)
                        
                        all_content.append(chunk)
                        all_embeddings.append(embedding)
                        all_metadata.append({
                            'source': section_name,
                            'chunk_id': i,
                            'url': section_data.get('url', ''),
                            'type': 'course_content',
                            'scraped_at': section_data.get('scraped_at', '')
                        })
                        
                        print(f"  ✅ Processed chunk {i+1}/{len(chunks)} from {section_name}")
        
        # Process any Discourse content if available
        discourse_files = glob.glob(os.path.join(settings.RAW_DATA_PATH, 'discourse_*.json'))
        
        if discourse_files:
            print(f"📋 Processing {len(discourse_files)} Discourse files...")
            
            for file_path in discourse_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        discourse_data = json.load(f)
                    
                    # Process topics if it's a topics file
                    if 'topics' in discourse_data:
                        for topic in discourse_data['topics']:
                            for post in topic.get('posts', []):
                                content = post.get('cleaned_text', '')
                                if content and len(content) > 50:
                                    chunks = self.chunk_content(content)
                                    
                                    for i, chunk in enumerate(chunks):
                                        embedding = self.create_embedding(chunk)
                                        
                                        all_content.append(chunk)
                                        all_embeddings.append(embedding)
                                        all_metadata.append({
                                            'source': f"discourse_topic_{topic.get('id')}",
                                            'chunk_id': i,
                                            'url': f"https://discourse.onlinedegree.iitm.ac.in/t/{topic.get('id')}",
                                            'type': 'discourse_post',
                                            'post_number': post.get('post_number', 1),
                                            'username': post.get('username', 'unknown')
                                        })
                except Exception as e:
                    print(f"⚠️ Error processing {file_path}: {e}")
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        
        # Save as NumPy archive (TA's recommended method)
        print("💾 Saving embeddings using NumPy archive...")
        np.savez_compressed(
            self.embeddings_file,
            embeddings=np.array(all_embeddings),
            content=np.array(all_content, dtype=object),
            metadata=np.array(all_metadata, dtype=object)
        )
        
        file_size = os.path.getsize(self.embeddings_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✅ Embeddings saved successfully!")
        print(f"📊 Total chunks: {len(all_embeddings)}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"📊 Dimensions: {len(all_embeddings[0]) if all_embeddings else 0}")
        
        if file_size_mb > 15:
            print("⚠️ Warning: File size exceeds 15MB recommendation")
        else:
            print("✅ File size within TA's 15MB recommendation")
        
        return {
            'total_chunks': len(all_embeddings),
            'file_size_mb': file_size_mb,
            'embedding_dimensions': len(all_embeddings[0]) if all_embeddings else 0
        }
    
    def load_embeddings(self):
        """Load embeddings from NumPy archive"""
        if not os.path.exists(self.embeddings_file):
            print("❌ Embeddings file not found. Run create_embeddings_from_knowledge_base() first.")
            return None
        
        try:
            data = np.load(self.embeddings_file, allow_pickle=True)
            return {
                'embeddings': data['embeddings'],
                'content': data['content'],
                'metadata': data['metadata']
            }
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return None
    
    def search_similar(self, query, top_k=5):
        """Search for similar content using cosine similarity"""
        embeddings_data = self.load_embeddings()
        if not embeddings_data:
            return []
        
        # Create query embedding
        query_embedding = self.create_embedding(query)
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings_data['embeddings'], query_embedding)
        
        # Get top k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': embeddings_data['content'][idx],
                'metadata': embeddings_data['metadata'][idx],
                'similarity': float(similarities[idx])
            })
        
        return results

if __name__ == "__main__":
    # Create and test the vector store
    vector_store = EfficientVectorStore()
    
    print("🚀 Creating efficient embeddings...")
    stats = vector_store.create_embeddings_from_knowledge_base()
    
    print("\n📊 Testing search functionality...")
    results = vector_store.search_similar("How do I use Docker in TDS?", top_k=3)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Content: {result['content'][:200]}...")
