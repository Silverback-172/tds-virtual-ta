import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # AIPipe Configuration (instead of direct OpenAI)
    AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
    OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1"  # For chat completions
    OPENAI_EMBEDDING_URL = "https://aipipe.org/openai/v1"  # For embeddings
    
    # Model Configuration
    CHAT_MODEL = "google/gemini-2.0-flash-lite-001"  # Cost-effective via OpenRouter
    EMBEDDING_MODEL = "text-embedding-3-small"  # Via OpenAI endpoint
    
    # Scraping Configuration
    DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
    TDS_COURSE_URL = "https://tds.s-anand.net"
    
    # Data paths
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    
    # Rate limiting
    REQUEST_DELAY = 1  # seconds between requests
    MAX_RETRIES = 3

settings = Settings()
