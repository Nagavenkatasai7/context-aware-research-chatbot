# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here  # Optional: for web search
TAVILY_API_KEY=your_tavily_key_here    # Optional: alternative web search

# Model Configuration
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TEMPERATURE=0.0

# Data Configuration
DATA_DIR=data
PDF_DIR=data/pdfs
VECTOR_STORE_DIR=data/vector_store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store Configuration
VECTOR_STORE_TYPE=faiss  # faiss or chroma
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# Memory Configuration
MEMORY_WINDOW=10

# Database Configuration
DATABASE_URL=sqlite:///./chatbot.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# UI Configuration
UI_TITLE=Context-Aware Research Chatbot
UI_DESCRIPTION=Ask questions about AI policy and get cited responses

# Evaluation Configuration
EVAL_DATASET_PATH=data/eval_dataset.json

# Logging
LOG_LEVEL=INFO