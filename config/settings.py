import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Gemini Configuration (Primary - Free for testing)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    
    # Model Configuration (Gemini models from session)
    EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model
    CHAT_MODEL = "gemini-2.0-flash"  # Gemini chat model (15 requests/min free)
    
    # Vector Storage Configuration
    EMBEDDINGS_FILE = PROCESSED_DATA_PATH / "comprehensive_embeddings.npz"
    MAX_EMBEDDINGS_SIZE_MB = 15

# Create settings instance
settings = Settings()

# Create directories if they don't exist
settings.RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
