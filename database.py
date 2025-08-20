"""
Database models and session management
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Generator

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import config

# Create database engine
engine = create_engine(config.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ChatSession(Base):
    """Chat session model"""
    __tablename__ = "chat_sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ChatSession(session_id='{self.session_id}', user_id='{self.user_id}')>"

class ConversationLog(Base):
    """Conversation log model"""
    __tablename__ = "conversation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_message = Column(Text)
    bot_response = Column(Text)
    tool_used = Column(String)
    sources = Column(JSON)  # Store sources as JSON
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ConversationLog(id={self.id}, session_id='{self.session_id}')>"

class EvaluationResult(Base):
    """Evaluation results model"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    expected_answer = Column(Text)
    actual_answer = Column(Text)
    tool_used = Column(String)
    sources = Column(JSON)
    faithfulness_score = Column(String)  # Store as string to handle different score types
    groundedness_score = Column(String)
    relevance_score = Column(String)
    evaluation_timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, question='{self.question[:50]}...')>"

# Create tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_session() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def get_session_stats(session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        db = SessionLocal()
        try:
            # Get session info
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            
            if not session:
                return {"error": "Session not found"}
            
            # Get conversation logs
            logs = db.query(ConversationLog).filter(
                ConversationLog.session_id == session_id
            ).all()
            
            # Calculate stats
            total_messages = len(logs)
            tools_used = {}
            
            for log in logs:
                tool = log.tool_used
                tools_used[tool] = tools_used.get(tool, 0) + 1
            
            return {
                "session_id": session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "total_messages": total_messages,
                "tools_used": tools_used
            }
            
        finally:
            db.close()
    
    @staticmethod
    def get_user_sessions(user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        db = SessionLocal()
        try:
            sessions = db.query(ChatSession).filter(
                ChatSession.user_id == user_id
            ).order_by(ChatSession.last_activity.desc()).all()
            
            return [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat()
                }
                for session in sessions
            ]
            
        finally:
            db.close()
    
    @staticmethod
    def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        db = SessionLocal()
        try:
            logs = db.query(ConversationLog).filter(
                ConversationLog.session_id == session_id
            ).order_by(ConversationLog.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": log.id,
                    "user_message": log.user_message,
                    "bot_response": log.bot_response,
                    "tool_used": log.tool_used,
                    "sources": log.sources,
                    "timestamp": log.timestamp.isoformat()
                }
                for log in reversed(logs)  # Reverse to get chronological order
            ]
            
        finally:
            db.close()
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a session and all its conversation logs"""
        db = SessionLocal()
        try:
            # Delete conversation logs first
            db.query(ConversationLog).filter(
                ConversationLog.session_id == session_id
            ).delete()
            
            # Delete session
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            
            if session:
                db.delete(session)
                db.commit()
                return True
            return False
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def cleanup_old_sessions(days: int = 30):
        """Clean up sessions older than specified days"""
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        db = SessionLocal()
        try:
            # Get old sessions
            old_sessions = db.query(ChatSession).filter(
                ChatSession.last_activity < cutoff_date
            ).all()
            
            session_ids = [session.session_id for session in old_sessions]
            
            # Delete conversation logs
            db.query(ConversationLog).filter(
                ConversationLog.session_id.in_(session_ids)
            ).delete(synchronize_session=False)
            
            # Delete sessions
            db.query(ChatSession).filter(
                ChatSession.last_activity < cutoff_date
            ).delete(synchronize_session=False)
            
            db.commit()
            
            return len(session_ids)
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    @staticmethod
    def get_global_stats() -> Dict[str, Any]:
        """Get global usage statistics"""
        db = SessionLocal()
        try:
            total_sessions = db.query(ChatSession).count()
            total_messages = db.query(ConversationLog).count()
            
            # Tool usage stats
            tool_stats = {}
            logs = db.query(ConversationLog.tool_used).all()
            for (tool,) in logs:
                tool_stats[tool] = tool_stats.get(tool, 0) + 1
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(days=1)
            recent_sessions = db.query(ChatSession).filter(
                ChatSession.last_activity > recent_cutoff
            ).count()
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "tool_usage": tool_stats,
                "recent_active_sessions": recent_sessions
            }
            
        finally:
            db.close()

# Initialize database
def init_database():
    """Initialize the database"""
    create_tables()
    print("Database initialized successfully")

if __name__ == "__main__":
    init_database()
    
    # Test database functionality
    db_manager = DatabaseManager()
    
    # Test creating a session
    db = SessionLocal()
    test_session = ChatSession(
        session_id="test-session-123",
        user_id="test-user"
    )
    db.add(test_session)
    db.commit()
    
    # Test conversation log
    test_log = ConversationLog(
        session_id="test-session-123",
        user_message="Test question",
        bot_response="Test response",
        tool_used="rag",
        sources=[{"source": "test.pdf", "page": 1}]
    )
    db.add(test_log)
    db.commit()
    db.close()
    
    # Test stats
    stats = db_manager.get_session_stats("test-session-123")
    print("Session stats:", stats)
    
    global_stats = db_manager.get_global_stats()
    print("Global stats:", global_stats)
    
    print("Database test completed successfully!")