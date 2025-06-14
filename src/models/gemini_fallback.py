import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import google.generativeai as genai
import hashlib
import numpy as np

class GeminiFallback:
    def __init__(self):
        # Configure Gemini (free for testing)
        genai.configure(api_key="your_gemini_api_key_here")  # Get from Google AI Studio
        self.model = genai.GenerativeModel('gemini-pro')
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
    
    def create_embedding(self, text):
        """Create deterministic embedding for testing (TA RECOMMENDED)"""
        # Use hash-based embedding for consistent testing
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding = np.array([int(text_hash[i:i+2], 16) / 255.0 for i in range(0, min(768, len(text_hash)), 2)])
        
        # Pad or truncate to exactly 384 dimensions
        if len(embedding) < 384:
            embedding = np.pad(embedding, (0, 384 - len(embedding)), 'constant')
        else:
            embedding = embedding[:384]
            
        return embedding
    
    def process_image(self, image_base64):
        """Process image using Gemini Vision (TA METHOD)"""
        try:
            prompt = """
            Analyze this image from a Tools in Data Science course discussion.
            
            Provide a comprehensive description focusing on:
            1. Any text visible (error messages, code, commands, UI elements)
            2. Technical content (screenshots, diagrams, code snippets)
            3. Context for TDS questions about Docker, Git, Python, data science
            4. Step-by-step processes or instructions visible
            5. Any error messages or configuration screens
            
            Be detailed and technical - this will help answer student questions.
            """
            
            # Convert base64 to image for Gemini
            import base64
            from PIL import Image
            import io
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            response = self.vision_model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            print(f"Gemini image processing failed: {e}")
            return "Image provided but could not be processed with Gemini fallback."
    
    def generate_response(self, question, context, sources):
        """Generate response using Gemini (TA RECOMMENDED FOR TESTING)"""
        try:
            prompt = f"""
            You are a Teaching Assistant for Tools in Data Science (TDS) course at IIT Madras.
            
            Student question: {question}
            
            Context from course materials:
            {context[:2000]}
            
            Provide a helpful, accurate response that:
            1. Directly answers the question
            2. Uses the course material context
            3. Is clear and educational
            4. Includes practical examples when relevant
            
            Keep response concise but comprehensive.
            """
            
            response = self.model.generate_content(prompt)
            return {
                "answer": response.text,
                "enhanced": True
            }
            
        except Exception as e:
            print(f"Gemini response generation failed: {e}")
            return {
                "answer": context[:500] + "...",
                "enhanced": False
            }

if __name__ == "__main__":
    # Test Gemini fallback
    fallback = GeminiFallback()
    test_response = fallback.generate_response(
        "How do I use Docker in TDS?",
        "Docker is a containerization tool...",
        ["docker section"]
    )
    print(test_response)
