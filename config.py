"""
Configuration module for the Context-Aware Research Chatbot
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class AppConfig(BaseSettings):
    """Application configuration"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    serpapi_key: Optional[str] = Field(None, env="SERPAPI_API_KEY")
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")
    
    # Model Configuration
    llm_model: str = Field("gpt-4o", env="LLM_MODEL")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    temperature: float = Field(0.0, env="TEMPERATURE")
    
    # Data Configuration
    data_dir: Path = Field(Path("data"), env="DATA_DIR")
    pdf_dir: Path = Field(Path("data/pdfs"), env="PDF_DIR")
    vector_store_dir: Path = Field(Path("data/vector_store"), env="VECTOR_STORE_DIR")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Vector Store Configuration
    vector_store_type: str = Field("faiss", env="VECTOR_STORE_TYPE")  # faiss or chroma
    top_k_retrieval: int = Field(5, env="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    
    # Memory Configuration
    memory_window: int = Field(10, env="MEMORY_WINDOW")
    
    # Database
    database_url: str = Field("sqlite:///./chatbot.db", env="DATABASE_URL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # UI Configuration
    ui_title: str = Field("Context-Aware Research Chatbot", env="UI_TITLE")
    ui_description: str = Field("Ask questions about AI policy and get cited responses", env="UI_DESCRIPTION")
    
    # Evaluation Configuration
    eval_dataset_path: Path = Field(Path("data/eval_dataset.json"), env="EVAL_DATASET_PATH")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields

    def __post_init__(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.pdf_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)

# Create global config instance
config = AppConfig()

# Prompt templates
PROMPTS = {
    "router": """
    You are a query router. Given a user query, determine which tool should handle it.
    
    Available tools:
    - rag: For questions about AI policy, regulations, or domain-specific knowledge
    - web_search: For current events, recent news, or information not in the knowledge base
    - math: For mathematical calculations or numerical problems
    
    Examples:
    Query: "What are the latest AI safety guidelines?"
    Answer: rag
    
    Query: "What happened in AI news today?"
    Answer: web_search
    
    Query: "Calculate 15% of 250,000"
    Answer: math
    
    Query: "What is GDPR's stance on AI?"
    Answer: rag
    
    Query: {query}
    Answer:
    """,
    
    "rag_qa": """
    You are an AI policy expert. Answer the user's question based on the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    1. Provide a comprehensive answer based on the context
    2. If the context doesn't contain enough information, say so
    3. Always cite your sources using [Source: filename, page/section]
    4. Be precise and factual
    
    Answer:
    """,
    
    "web_search_qa": """
    You are a helpful research assistant. Answer the user's question based on the web search results.
    
    Search Results:
    {search_results}
    
    Question: {question}
    
    Instructions:
    1. Provide a comprehensive answer based on the search results
    2. Cite sources with URLs when available
    3. Be current and factual
    4. If results are insufficient, mention it
    
    Answer:
    """
}