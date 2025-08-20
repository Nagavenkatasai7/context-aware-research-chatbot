"""
FastAPI backend for the Context-Aware Research Chatbot
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Local imports
from config import config
from chatbot import get_chatbot, ContextAwareChatbot
from database import get_session, DatabaseManager, ChatSession, ConversationLog, init_database
from sqlalchemy.orm import Session

# Initialize logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

# Initialize database
init_database()

# Create FastAPI app
app = FastAPI(
    title="Context-Aware Research Chatbot API",
    description="A conversational agent that answers domain questions using web search, local RAG, and citations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    tool_used: str
    routing_explanation: str
    session_id: str
    timestamp: datetime

class SessionRequest(BaseModel):
    user_id: str = "anonymous"

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime

class SessionStats(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    last_activity: str
    total_messages: int
    tools_used: Dict[str, int]

class GlobalStats(BaseModel):
    total_sessions: int
    total_messages: int
    tool_usage: Dict[str, int]
    recent_active_sessions: int

class ConversationHistory(BaseModel):
    history: List[Dict[str, Any]]

# Dependency to get chatbot instance
def get_chatbot_instance() -> ContextAwareChatbot:
    return get_chatbot()

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Context-Aware Research Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "features": [
            "Web search integration",
            "Local knowledge base (RAG)",
            "Mathematical calculations",
            "Conversational memory",
            "Source citations",
            "Session management"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        chatbot = get_chatbot_instance()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "rag_available": chatbot.rag_tool is not None,
            "web_search_available": (chatbot.web_search.serp_api is not None or 
                                   chatbot.web_search.tavily_client is not None)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {e}")

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest, chatbot: ContextAwareChatbot = Depends(get_chatbot_instance)):
    """Create a new chat session"""
    try:
        session_id = chatbot.create_session(request.user_id)
        return SessionResponse(
            session_id=session_id,
            created_at=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, chatbot: ContextAwareChatbot = Depends(get_chatbot_instance)):
    """Chat with the bot"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = chatbot.create_session(request.user_id)
        
        # Process the message
        result = chatbot.chat(request.message, session_id)
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            tool_used=result["tool_used"],
            routing_explanation=result["routing_explanation"],
            session_id=result["session_id"],
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.get("/sessions/{session_id}/history", response_model=ConversationHistory)
async def get_session_history(session_id: str, chatbot: ContextAwareChatbot = Depends(get_chatbot_instance)):
    """Get conversation history for a session"""
    try:
        history = chatbot.get_session_history(session_id)
        return ConversationHistory(history=history)
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {e}")

@app.get("/sessions/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(session_id: str):
    """Get statistics for a session"""
    try:
        stats = DatabaseManager.get_session_stats(session_id)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return SessionStats(**stats)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, chatbot: ContextAwareChatbot = Depends(get_chatbot_instance)):
    """Delete a session and its history"""
    try:
        success = DatabaseManager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Clear from memory
        chatbot.clear_session(session_id)
        
        return {"message": f"Session {session_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {e}")

@app.get("/sessions/{session_id}/clear")
async def clear_session_memory(session_id: str, chatbot: ContextAwareChatbot = Depends(get_chatbot_instance)):
    """Clear session memory (but keep database history)"""
    try:
        chatbot.clear_session(session_id)
        return {"message": f"Session {session_id} memory cleared"}
    except Exception as e:
        logger.error(f"Error clearing session memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {e}")

@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    try:
        sessions = DatabaseManager.get_user_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user sessions: {e}")

@app.get("/stats", response_model=GlobalStats)
async def get_global_stats():
    """Get global usage statistics"""
    try:
        stats = DatabaseManager.get_global_stats()
        return GlobalStats(**stats)
    except Exception as e:
        logger.error(f"Error getting global stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

@app.post("/admin/cleanup")
async def cleanup_old_sessions(days: int = 30, background_tasks: BackgroundTasks):
    """Clean up old sessions (admin endpoint)"""
    try:
        def cleanup_task():
            count = DatabaseManager.cleanup_old_sessions(days)
            logger.info(f"Cleaned up {count} old sessions")
        
        background_tasks.add_task(cleanup_task)
        return {"message": f"Cleanup task started for sessions older than {days} days"}
    except Exception as e:
        logger.error(f"Error starting cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start cleanup: {e}")

@app.get("/config")
async def get_config():
    """Get non-sensitive configuration information"""
    return {
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "vector_store_type": config.vector_store_type,
        "top_k_retrieval": config.top_k_retrieval,
        "memory_window": config.memory_window,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    logger.info("Starting Context-Aware Research Chatbot API")
    try:
        chatbot = get_chatbot_instance()
        logger.info("✅ Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Context-Aware Research Chatbot API")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
        log_level=config.log_level.lower()
    )