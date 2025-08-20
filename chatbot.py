"""
Core chatbot module with memory, routing, and conversation management
"""
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# LangChain imports
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

# Local imports
from config import config, PROMPTS
from tools import QueryRouter, WebSearchTool, MathTool, RAGTool
from database import ChatSession, ConversationLog, get_session
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextAwareChatbot:
    """Main chatbot class with context awareness and tool routing"""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key
        )
        
        # Initialize components
        self.router = QueryRouter()
        self.web_search = WebSearchTool()
        self.math_tool = MathTool()
        
        # Initialize vector store and RAG
        self.vector_store = None
        self.retriever = None
        self.rag_tool = None
        self._initialize_rag()
        
        # Memory storage for sessions
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        
        # Initialize prompt templates
        self._initialize_prompts()
        
        logger.info("Chatbot initialized successfully")
    
    def _initialize_rag(self):
        """Initialize RAG components"""
        try:
            processor = DataProcessor()
            self.vector_store = processor.load_vector_store()
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": config.top_k_retrieval}
            )
            self.rag_tool = RAGTool(self.retriever)
            logger.info("RAG components initialized")
        except Exception as e:
            logger.warning(f"Could not initialize RAG: {e}")
            logger.info("Chatbot will work without local knowledge base")
    
    def _initialize_prompts(self):
        """Initialize prompt templates"""
        self.router_prompt = PromptTemplate(
            input_variables=["query"],
            template=PROMPTS["router"]
        )
        
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPTS["rag_qa"]
        )
        
        self.web_search_prompt = PromptTemplate(
            input_variables=["search_results", "question"],
            template=PROMPTS["web_search_qa"]
        )
    
    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a session"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=config.memory_window,
                return_messages=True,
                memory_key="chat_history"
            )
        return self.memories[session_id]
    
    def create_session(self, user_id: str = "anonymous") -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        
        # Store in database
        db = next(get_session())
        try:
            session = ChatSession(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            logger.info(f"Created new session: {session_id}")
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            db.rollback()
        finally:
            db.close()
        
        return session_id
    
    def _log_conversation(self, session_id: str, user_message: str, 
                         bot_response: str, tool_used: str, sources: List[Dict]):
        """Log conversation to database"""
        db = next(get_session())
        try:
            log_entry = ConversationLog(
                session_id=session_id,
                user_message=user_message,
                bot_response=bot_response,
                tool_used=tool_used,
                sources=sources,
                timestamp=datetime.utcnow()
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _handle_rag_query(self, query: str, memory: ConversationBufferWindowMemory) -> Tuple[str, List[Dict]]:
        """Handle RAG-based queries"""
        if not self.rag_tool:
            return "Knowledge base is not available. Please try a web search instead.", []
        
        try:
            # Get relevant documents
            rag_result = self.rag_tool.query(query)
            context = rag_result['context']
            sources = rag_result['sources']
            
            # Generate response using context
            chain = LLMChain(llm=self.llm, prompt=self.rag_prompt)
            response = chain.run(context=context, question=query)
            
            # Add to memory
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response)
            
            return response, sources
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return f"Error processing knowledge base query: {e}", []
    
    def _handle_web_search_query(self, query: str, memory: ConversationBufferWindowMemory) -> Tuple[str, List[Dict]]:
        """Handle web search queries"""
        try:
            # Perform web search
            search_results = self.web_search.search(query, num_results=5)
            
            # Generate response using search results
            chain = LLMChain(llm=self.llm, prompt=self.web_search_prompt)
            response = chain.run(search_results=search_results, question=query)
            
            # Add to memory
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response)
            
            # Create sources (simplified for web search)
            sources = [{"source": "web_search", "query": query}]
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error performing web search: {e}", []
    
    def _handle_math_query(self, query: str, memory: ConversationBufferWindowMemory) -> Tuple[str, List[Dict]]:
        """Handle math queries"""
        try:
            # Extract mathematical expression
            import re
            math_expr = re.search(r'[\d\+\-\*/\(\)\.\s]+', query)
            if math_expr:
                expression = math_expr.group().strip()
                result = self.math_tool.calculate(expression)
                
                # Add context to the response
                response = f"Mathematical calculation for: {expression}\n{result}"
            else:
                response = self.math_tool.calculate(query)
            
            # Add to memory
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response)
            
            sources = [{"source": "calculator", "expression": query}]
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Math query error: {e}")
            return f"Error performing calculation: {e}", []
    
    def chat(self, message: str, session_id: str) -> Dict[str, Any]:
        """Main chat method"""
        logger.info(f"Processing message in session {session_id}: {message[:100]}...")
        
        # Get memory for this session
        memory = self.get_memory(session_id)
        
        # Route the query
        route = self.router.route(message)
        logger.info(f"Routed to: {route}")
        
        # Handle based on route
        if route == "rag":
            response, sources = self._handle_rag_query(message, memory)
        elif route == "web_search":
            response, sources = self._handle_web_search_query(message, memory)
        elif route == "math":
            response, sources = self._handle_math_query(message, memory)
        else:
            # Fallback to RAG
            response, sources = self._handle_rag_query(message, memory)
        
        # Log conversation
        self._log_conversation(session_id, message, response, route, sources)
        
        # Get routing explanation
        routing_explanation = self.router.get_routing_explanation(message)
        
        result = {
            "response": response,
            "sources": sources,
            "tool_used": route,
            "routing_explanation": routing_explanation,
            "session_id": session_id
        }
        
        logger.info(f"Response generated successfully using {route}")
        return result
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        memory = self.get_memory(session_id)
        history = []
        
        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_session(self, session_id: str):
        """Clear memory for a session"""
        if session_id in self.memories:
            del self.memories[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        db = next(get_session())
        try:
            logs = db.query(ConversationLog).filter(
                ConversationLog.session_id == session_id
            ).all()
            
            total_messages = len(logs)
            tools_used = {}
            for log in logs:
                tool = log.tool_used
                tools_used[tool] = tools_used.get(tool, 0) + 1
            
            return {
                "total_messages": total_messages,
                "tools_used": tools_used,
                "session_id": session_id
            }
        finally:
            db.close()

# Global chatbot instance
chatbot = None

def get_chatbot() -> ContextAwareChatbot:
    """Get or create chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = ContextAwareChatbot()
    return chatbot

def test_chatbot():
    """Test the chatbot functionality"""
    bot = get_chatbot()
    session_id = bot.create_session("test_user")
    
    test_queries = [
        "What is GDPR?",
        "Calculate 15% of 250000",
        "Latest AI safety news"
    ]
    
    for query in test_queries:
        print(f"\n🤖 Testing: {query}")
        result = bot.chat(query, session_id)
        print(f"Tool used: {result['tool_used']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Sources: {len(result['sources'])}")

if __name__ == "__main__":
    test_chatbot()