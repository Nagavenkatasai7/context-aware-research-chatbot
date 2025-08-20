"""
Tools module for web search and calculations
"""
import logging
import re
from typing import List, Dict, Any, Optional
from langchain.tools import Tool, BaseTool
from langchain_community.utilities import SerpAPIWrapper
from langchain.schema import BaseRetriever

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search tool using SerpAPI or Tavily"""
    
    def __init__(self):
        self.serp_api = None
        self.tavily_client = None
        
        # Initialize SerpAPI if key is available
        if config.serpapi_key:
            try:
                self.serp_api = SerpAPIWrapper(serpapi_api_key=config.serpapi_key)
                logger.info("SerpAPI initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SerpAPI: {e}")
        
        # Initialize Tavily if key is available
        if config.tavily_api_key and TavilyClient:
            try:
                self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
                logger.info("Tavily initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily: {e}")
    
    def search(self, query: str, num_results: int = 5) -> str:
        """Perform web search"""
        if self.tavily_client:
            return self._search_tavily(query, num_results)
        elif self.serp_api:
            return self._search_serpapi(query)
        else:
            return "Web search is not available. Please configure SERPAPI_API_KEY or TAVILY_API_KEY."
    
    def _search_tavily(self, query: str, num_results: int = 5) -> str:
        """Search using Tavily"""
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results
            )
            
            results = []
            for result in response.get('results', []):
                result_text = f"Title: {result.get('title', 'N/A')}\n"
                result_text += f"URL: {result.get('url', 'N/A')}\n"
                result_text += f"Content: {result.get('content', 'N/A')}\n"
                results.append(result_text)
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return f"Search failed: {e}"
    
    def _search_serpapi(self, query: str) -> str:
        """Search using SerpAPI"""
        try:
            return self.serp_api.run(query)
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return f"Search failed: {e}"

class MathTool:
    """Math calculation tool"""
    
    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Allow only safe mathematical operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Only basic mathematical operations are allowed (+, -, *, /, parentheses)"
            
            # Evaluate the expression
            result = eval(expression)
            return f"Result: {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"

class RAGTool:
    """RAG tool for querying the vector store"""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the vector store and return relevant documents"""
        try:
            docs = self.retriever.get_relevant_documents(question)
            
            # Format the context
            context_parts = []
            sources = []
            
            for i, doc in enumerate(docs):
                context_parts.append(f"[Document {i+1}]: {doc.page_content}")
                
                # Extract source information
                source_info = {
                    'file': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', i)
                }
                sources.append(source_info)
            
            return {
                'context': '\n\n'.join(context_parts),
                'sources': sources,
                'num_sources': len(docs)
            }
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return {
                'context': f"Error retrieving documents: {e}",
                'sources': [],
                'num_sources': 0
            }

def create_tools(retriever: Optional[BaseRetriever] = None) -> List[Tool]:
    """Create all tools for the chatbot"""
    tools = []
    
    # Web search tool
    web_search = WebSearchTool()
    web_search_tool = Tool(
        name="web_search",
        description="Search the web for current information, news, and recent developments",
        func=web_search.search
    )
    tools.append(web_search_tool)
    
    # Math tool
    math_tool = Tool(
        name="calculator",
        description="Perform mathematical calculations. Input should be a mathematical expression.",
        func=MathTool.calculate
    )
    tools.append(math_tool)
    
    # RAG tool (if retriever is provided)
    if retriever:
        rag_tool_instance = RAGTool(retriever)
        rag_tool = Tool(
            name="knowledge_base",
            description="Search the local knowledge base for domain-specific information",
            func=lambda q: rag_tool_instance.query(q)['context']
        )
        tools.append(rag_tool)
    
    return tools

class QueryRouter:
    """Routes queries to appropriate tools"""
    
    def __init__(self):
        self.math_keywords = ['calculate', 'compute', 'math', 'arithmetic', '+', '-', '*', '/', '%']
        self.web_keywords = ['latest', 'recent', 'news', 'current', 'today', 'yesterday', 'breaking']
        self.rag_keywords = ['policy', 'regulation', 'guideline', 'law', 'compliance', 'standard']
    
    def route(self, query: str) -> str:
        """Determine which tool should handle the query"""
        query_lower = query.lower()
        
        # Check for math expressions
        if any(keyword in query_lower for keyword in self.math_keywords):
            # Also check if it looks like a math expression
            if re.search(r'\d+\s*[+\-*/]\s*\d+', query):
                return "math"
        
        # Check for web search indicators
        if any(keyword in query_lower for keyword in self.web_keywords):
            return "web_search"
        
        # Check for RAG indicators
        if any(keyword in query_lower for keyword in self.rag_keywords):
            return "rag"
        
        # Default to RAG for domain questions
        return "rag"
    
    def get_routing_explanation(self, query: str) -> str:
        """Get explanation for routing decision"""
        route = self.route(query)
        
        explanations = {
            "math": "Routing to calculator for mathematical computation",
            "web_search": "Routing to web search for current information",
            "rag": "Routing to knowledge base for domain-specific information"
        }
        
        return explanations.get(route, "Using default routing to knowledge base")

def test_tools():
    """Test function for tools"""
    print("Testing tools...")
    
    # Test math tool
    math_tool = MathTool()
    print(f"Math test: {math_tool.calculate('15 * 20 + 100')}")
    
    # Test web search
    web_search = WebSearchTool()
    if web_search.serp_api or web_search.tavily_client:
        result = web_search.search("AI policy news", 2)
        print(f"Web search test: {result[:200]}...")
    else:
        print("Web search not configured")
    
    # Test router
    router = QueryRouter()
    test_queries = [
        "What is 15% of 250000?",
        "Latest AI safety regulations",
        "What does GDPR say about AI?"
    ]
    
    for query in test_queries:
        route = router.route(query)
        explanation = router.get_routing_explanation(query)
        print(f"Query: '{query}' -> Route: {route} ({explanation})")

if __name__ == "__main__":
    test_tools()