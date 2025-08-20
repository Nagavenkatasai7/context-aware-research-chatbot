"""
Tests for tools functionality
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import WebSearchTool, MathTool, QueryRouter, RAGTool, create_tools
from config import config

class TestMathTool:
    """Test cases for MathTool"""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        math_tool = MathTool()
        
        test_cases = [
            ("2 + 2", "Result: 4"),
            ("10 - 5", "Result: 5"),
            ("3 * 4", "Result: 12"),
            ("15 / 3", "Result: 5.0"),
            ("2 ** 3", "Result: 8"),
            ("10 % 3", "Result: 1")
        ]
        
        for expression, expected in test_cases:
            result = math_tool.calculate(expression)
            assert expected in result, f"Expression '{expression}' should contain '{expected}', got '{result}'"
    
    def test_complex_expressions(self):
        """Test complex mathematical expressions"""
        math_tool = MathTool()
        
        test_cases = [
            ("(2 + 3) * 4", "Result: 20"),
            ("2 * (5 + 3)", "Result: 16"),
            ("(10 - 2) / 4", "Result: 2.0"),
            ("2 + 3 * 4 - 1", "Result: 13")
        ]
        
        for expression, expected in test_cases:
            result = math_tool.calculate(expression)
            assert expected in result, f"Expression '{expression}' should contain '{expected}', got '{result}'"
    
    def test_decimal_operations(self):
        """Test decimal number operations"""
        math_tool = MathTool()
        
        test_cases = [
            ("3.14 + 2.86", "Result: 6.0"),
            ("10.5 / 2", "Result: 5.25"),
            ("0.1 + 0.2", "Result: 0.30000000000000004")  # Floating point precision
        ]
        
        for expression, expected in test_cases:
            result = math_tool.calculate(expression)
            assert "Result:" in result, f"Expression '{expression}' should return a result"
    
    def test_error_handling(self):
        """Test error handling in math operations"""
        math_tool = MathTool()
        
        # Division by zero
        result = math_tool.calculate("10 / 0")
        assert "Error: Division by zero" in result
        
        # Invalid characters
        result = math_tool.calculate("2 + abc")
        assert "Error:" in result
        
        # Invalid expression
        result = math_tool.calculate("2 +")
        assert "Error:" in result
    
    def test_security_restrictions(self):
        """Test that dangerous operations are blocked"""
        math_tool = MathTool()
        
        # Test potentially dangerous expressions
        dangerous_expressions = [
            "import os",
            "__import__('os')",
            "eval('print(1)')",
            "exec('print(1)')"
        ]
        
        for expression in dangerous_expressions:
            result = math_tool.calculate(expression)
            assert "Error:" in result, f"Dangerous expression '{expression}' should be blocked"

class TestQueryRouter:
    """Test cases for QueryRouter"""
    
    def test_math_routing(self):
        """Test routing of math queries"""
        router = QueryRouter()
        
        math_queries = [
            "Calculate 15% of 250",
            "What is 2 + 2?",
            "Compute 15 * 20",
            "What's the result of 100 / 4?",
            "Find the square root of 16"
        ]
        
        for query in math_queries:
            route = router.route(query)
            assert route == "math", f"Query '{query}' should route to 'math', got '{route}'"
    
    def test_web_search_routing(self):
        """Test routing of web search queries"""
        router = QueryRouter()
        
        web_queries = [
            "Latest AI news today",
            "Recent developments in machine learning",
            "What happened in tech news yesterday?",
            "Current AI safety guidelines",
            "Breaking news in artificial intelligence"
        ]
        
        for query in web_queries:
            route = router.route(query)
            assert route == "web_search", f"Query '{query}' should route to 'web_search', got '{route}'"
    
    def test_rag_routing(self):
        """Test routing of RAG/knowledge base queries"""
        router = QueryRouter()
        
        rag_queries = [
            "What is GDPR?",
            "AI policy guidelines",
            "Privacy regulations for AI",
            "What does the AI Act say about compliance?",
            "Data protection standards"
        ]
        
        for query in rag_queries:
            route = router.route(query)
            assert route == "rag", f"Query '{query}' should route to 'rag', got '{route}'"
    
    def test_default_routing(self):
        """Test default routing behavior"""
        router = QueryRouter()
        
        # Ambiguous queries should default to RAG
        ambiguous_queries = [
            "Tell me about artificial intelligence",
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?"
        ]
        
        for query in ambiguous_queries:
            route = router.route(query)
            assert route == "rag", f"Ambiguous query '{query}' should default to 'rag', got '{route}'"
    
    def test_routing_explanation(self):
        """Test routing explanation functionality"""
        router = QueryRouter()
        
        test_cases = [
            ("Calculate 2+2", "math"),
            ("Latest news", "web_search"),
            ("What is GDPR?", "rag")
        ]
        
        for query, expected_route in test_cases:
            explanation = router.get_routing_explanation(query)
            assert isinstance(explanation, str), "Explanation should be a string"
            assert len(explanation) > 0, "Explanation should not be empty"

class TestWebSearchTool:
    """Test cases for WebSearchTool"""
    
    def test_initialization_without_keys(self):
        """Test WebSearchTool initialization without API keys"""
        # Temporarily clear API keys
        original_serpapi = config.serpapi_key
        original_tavily = config.tavily_api_key
        
        config.serpapi_key = None
        config.tavily_api_key = None
        
        try:
            web_search = WebSearchTool()
            assert web_search.serp_api is None
            assert web_search.tavily_client is None
        finally:
            # Restore original keys
            config.serpapi_key = original_serpapi
            config.tavily_api_key = original_tavily
    
    def test_search_without_apis(self):
        """Test search functionality without configured APIs"""
        # Create WebSearchTool without API keys
        web_search = WebSearchTool()
        web_search.serp_api = None
        web_search.tavily_client = None
        
        result = web_search.search("test query")
        assert "Web search is not available" in result
    
    @patch('tools.SerpAPIWrapper')
    def test_serpapi_search(self, mock_serpapi):
        """Test SerpAPI search functionality"""
        # Mock SerpAPI
        mock_serpapi_instance = Mock()
        mock_serpapi_instance.run.return_value = "Sample search results"
        mock_serpapi.return_value = mock_serpapi_instance
        
        # Set up WebSearchTool with mocked SerpAPI
        config.serpapi_key = "test-key"
        web_search = WebSearchTool()
        
        result = web_search.search("test query")
        assert result == "Sample search results"
        mock_serpapi_instance.run.assert_called_once_with("test query")
    
    @patch('tools.TavilyClient')
    def test_tavily_search(self, mock_tavily):
        """Test Tavily search functionality"""
        # Mock Tavily client
        mock_tavily_instance = Mock()
        mock_tavily_response = {
            'results': [
                {
                    'title': 'Test Title',
                    'url': 'https://example.com',
                    'content': 'Test content'
                }
            ]
        }
        mock_tavily_instance.search.return_value = mock_tavily_response
        mock_tavily.return_value = mock_tavily_instance
        
        # Set up WebSearchTool with mocked Tavily
        config.tavily_api_key = "test-key"
        web_search = WebSearchTool()
        web_search.tavily_client = mock_tavily_instance
        
        result = web_search.search("test query")
        
        assert "Test Title" in result
        assert "https://example.com" in result
        assert "Test content" in result

class TestRAGTool:
    """Test cases for RAGTool"""
    
    def test_rag_tool_initialization(self):
        """Test RAGTool initialization"""
        mock_retriever = Mock()
        rag_tool = RAGTool(mock_retriever)
        
        assert rag_tool.retriever is mock_retriever
    
    def test_rag_query(self):
        """Test RAG query functionality"""
        # Mock retriever and documents
        mock_retriever = Mock()
        mock_doc1 = Mock()
        mock_doc1.page_content = "Sample content 1"
        mock_doc1.metadata = {
            'source_file': 'test1.pdf',
            'page': 1,
            'chunk_id': 0
        }
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Sample content 2"
        mock_doc2.metadata = {
            'source_file': 'test2.pdf',
            'page': 2,
            'chunk_id': 1
        }
        
        mock_retriever.get_relevant_documents.return_value = [mock_doc1, mock_doc2]
        
        rag_tool = RAGTool(mock_retriever)
        result = rag_tool.query("test question")
        
        assert result['num_sources'] == 2
        assert "Sample content 1" in result['context']
        assert "Sample content 2" in result['context']
        assert len(result['sources']) == 2
        assert result['sources'][0]['file'] == 'test1.pdf'
        assert result['sources'][1]['file'] == 'test2.pdf'
    
    def test_rag_query_error_handling(self):
        """Test RAG query error handling"""
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.side_effect = Exception("Retrieval error")
        
        rag_tool = RAGTool(mock_retriever)
        result = rag_tool.query("test question")
        
        assert "Error retrieving documents" in result['context']
        assert result['num_sources'] == 0
        assert len(result['sources']) == 0

class TestToolIntegration:
    """Integration tests for tools"""
    
    def test_create_tools_without_retriever(self):
        """Test creating tools without retriever"""
        tools = create_tools()
        
        # Should have web search and math tools
        tool_names = [tool.name for tool in tools]
        assert "web_search" in tool_names
        assert "calculator" in tool_names
        assert "knowledge_base" not in tool_names  # Should not have RAG tool
    
    def test_create_tools_with_retriever(self):
        """Test creating tools with retriever"""
        mock_retriever = Mock()
        tools = create_tools(mock_retriever)
        
        # Should have all three tools
        tool_names = [tool.name for tool in tools]
        assert "web_search" in tool_names
        assert "calculator" in tool_names
        assert "knowledge_base" in tool_names
    
    def test_tool_descriptions(self):
        """Test that all tools have proper descriptions"""
        mock_retriever = Mock()
        tools = create_tools(mock_retriever)
        
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert len(tool.name) > 0
            assert len(tool.description) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])