"""
Tests for the chatbot functionality
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot import ContextAwareChatbot, get_chatbot
from config import config
from database import init_database, get_session, ChatSession, ConversationLog

class TestContextAwareChatbot:
    """Test cases for ContextAwareChatbot"""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment"""
        # Use test database
        config.database_url = "sqlite:///./test_chatbot.db"
        init_database()
        
        # Mock OpenAI API key if not set
        if not config.openai_api_key:
            config.openai_api_key = "test-key"
        
        yield
        
        # Cleanup
        test_db_path = Path("test_chatbot.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    @patch('chatbot.ChatOpenAI')
    @patch('chatbot.DataProcessor')
    def test_chatbot_initialization(self, mock_processor, mock_llm):
        """Test chatbot initialization"""
        mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
        mock_llm.return_value = Mock()
        
        chatbot = ContextAwareChatbot()
        
        assert chatbot is not None
        assert chatbot.router is not None
        assert chatbot.web_search is not None
        assert chatbot.math_tool is not None
        assert chatbot.memories == {}
    
    def test_session_creation(self):
        """Test session creation"""
        with patch('chatbot.ChatOpenAI'), patch('chatbot.DataProcessor') as mock_processor:
            mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
            
            chatbot = ContextAwareChatbot()
            session_id = chatbot.create_session("test_user")
            
            assert session_id is not None
            assert len(session_id) > 0
            
            # Check if session exists in database
            db = next(get_session())
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id
            ).first()
            assert session is not None
            assert session.user_id == "test_user"
            db.close()
    
    def test_memory_management(self):
        """Test conversation memory management"""
        with patch('chatbot.ChatOpenAI'), patch('chatbot.DataProcessor') as mock_processor:
            mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
            
            chatbot = ContextAwareChatbot()
            session_id = "test-session"
            
            # Get memory for session
            memory1 = chatbot.get_memory(session_id)
            memory2 = chatbot.get_memory(session_id)
            
            # Should return the same memory instance
            assert memory1 is memory2
            assert session_id in chatbot.memories
    
    @patch('chatbot.ChatOpenAI')
    @patch('chatbot.DataProcessor')
    def test_math_query_handling(self, mock_processor, mock_llm):
        """Test math query handling"""
        mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        chatbot = ContextAwareChatbot()
        session_id = chatbot.create_session("test_user")
        
        # Test math query
        query = "Calculate 2 + 2"
        memory = chatbot.get_memory(session_id)
        
        response, sources = chatbot._handle_math_query(query, memory)
        
        assert "4" in response or "Result: 4" in response
        assert len(sources) > 0
        assert sources[0]["source"] == "calculator"
    
    @patch('chatbot.ChatOpenAI')
    @patch('chatbot.DataProcessor')
    @patch('chatbot.WebSearchTool')
    def test_web_search_query_handling(self, mock_web_search, mock_processor, mock_llm):
        """Test web search query handling"""
        mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        # Mock web search
        mock_web_search_instance = Mock()
        mock_web_search_instance.search.return_value = "Sample search results"
        
        # Mock LLM chain
        with patch('chatbot.LLMChain') as mock_chain:
            mock_chain_instance = Mock()
            mock_chain_instance.run.return_value = "Sample response based on search"
            mock_chain.return_value = mock_chain_instance
            
            chatbot = ContextAwareChatbot()
            chatbot.web_search = mock_web_search_instance
            
            session_id = chatbot.create_session("test_user")
            query = "Latest AI news"
            memory = chatbot.get_memory(session_id)
            
            response, sources = chatbot._handle_web_search_query(query, memory)
            
            assert response == "Sample response based on search"
            assert len(sources) > 0
            assert sources[0]["source"] == "web_search"
    
    @patch('chatbot.ChatOpenAI')
    @patch('chatbot.DataProcessor')
    def test_query_routing(self, mock_processor, mock_llm):
        """Test query routing logic"""
        mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
        mock_llm.return_value = Mock()
        
        chatbot = ContextAwareChatbot()
        
        # Test different query types
        test_cases = [
            ("Calculate 15% of 250", "math"),
            ("What is 2 + 2?", "math"),
            ("Latest AI news today", "web_search"),
            ("Recent developments in AI", "web_search"),
            ("What is GDPR?", "rag"),
            ("AI policy guidelines", "rag")
        ]
        
        for query, expected_route in test_cases:
            actual_route = chatbot.router.route(query)
            assert actual_route == expected_route, f"Query '{query}' should route to '{expected_route}', got '{actual_route}'"
    
    def test_session_stats(self):
        """Test session statistics"""
        with patch('chatbot.ChatOpenAI'), patch('chatbot.DataProcessor') as mock_processor:
            mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
            
            chatbot = ContextAwareChatbot()
            session_id = chatbot.create_session("test_user")
            
            # Add some fake conversation logs
            db = next(get_session())
            log1 = ConversationLog(
                session_id=session_id,
                user_message="Test message 1",
                bot_response="Test response 1",
                tool_used="rag",
                sources=[]
            )
            log2 = ConversationLog(
                session_id=session_id,
                user_message="Test message 2", 
                bot_response="Test response 2",
                tool_used="math",
                sources=[]
            )
            db.add(log1)
            db.add(log2)
            db.commit()
            db.close()
            
            # Get stats
            stats = chatbot.get_session_stats(session_id)
            
            assert stats["total_messages"] == 2
            assert stats["tools_used"]["rag"] == 1
            assert stats["tools_used"]["math"] == 1
            assert stats["session_id"] == session_id
    
    def test_clear_session(self):
        """Test session clearing"""
        with patch('chatbot.ChatOpenAI'), patch('chatbot.DataProcessor') as mock_processor:
            mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
            
            chatbot = ContextAwareChatbot()
            session_id = "test-session"
            
            # Create memory
            memory = chatbot.get_memory(session_id)
            assert session_id in chatbot.memories
            
            # Clear session
            chatbot.clear_session(session_id)
            assert session_id not in chatbot.memories
    
    def test_get_chatbot_singleton(self):
        """Test chatbot singleton pattern"""
        with patch('chatbot.ContextAwareChatbot') as mock_chatbot:
            mock_instance = Mock()
            mock_chatbot.return_value = mock_instance
            
            # Reset global chatbot
            import chatbot
            chatbot.chatbot = None
            
            bot1 = get_chatbot()
            bot2 = get_chatbot()
            
            # Should return same instance
            assert bot1 is bot2
            assert mock_chatbot.call_count == 1

class TestChatbotIntegration:
    """Integration tests for chatbot"""
    
    @pytest.fixture(autouse=True)
    def setup_integration_env(self):
        """Setup integration test environment"""
        config.database_url = "sqlite:///./test_integration.db"
        init_database()
        
        yield
        
        # Cleanup
        test_db_path = Path("test_integration.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    @patch('chatbot.ChatOpenAI')
    @patch('chatbot.DataProcessor')
    def test_full_chat_flow(self, mock_processor, mock_llm):
        """Test complete chat flow"""
        mock_processor.return_value.load_vector_store.side_effect = Exception("No vector store")
        
        # Mock LLM responses
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        with patch('chatbot.LLMChain') as mock_chain:
            mock_chain_instance = Mock()
            mock_chain_instance.run.return_value = "Test response"
            mock_chain.return_value = mock_chain_instance
            
            chatbot = ContextAwareChatbot()
            session_id = chatbot.create_session("integration_test_user")
            
            # Test math query
            result = chatbot.chat("What is 5 + 3?", session_id)
            
            assert "response" in result
            assert "sources" in result
            assert "tool_used" in result
            assert result["tool_used"] == "math"
            assert result["session_id"] == session_id
            
            # Check if conversation was logged
            db = next(get_session())
            logs = db.query(ConversationLog).filter(
                ConversationLog.session_id == session_id
            ).all()
            assert len(logs) == 1
            assert logs[0].tool_used == "math"
            db.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])