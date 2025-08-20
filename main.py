"""
Main entry point for the Context-Aware Research Chatbot
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Local imports
from config import config
from database import init_database
from data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project():
    """Initial project setup"""
    logger.info("🚀 Setting up Context-Aware Research Chatbot...")
    
    # Check if .env file exists
    if not Path(".env").exists():
        logger.warning("⚠️  .env file not found. Creating from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("📝 Please edit .env file with your API keys")
        else:
            logger.error("❌ .env.example not found")
            return False
    
    # Create necessary directories
    config.data_dir.mkdir(exist_ok=True)
    config.pdf_dir.mkdir(exist_ok=True)
    config.vector_store_dir.mkdir(exist_ok=True)
    
    # Initialize database
    try:
        init_database()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    logger.info("✅ Project setup completed!")
    return True

def process_pdfs():
    """Process PDF files and create vector store"""
    logger.info("📚 Processing PDF files...")
    
    # Check if PDFs exist
    pdf_files = list(config.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️  No PDF files found in {config.pdf_dir}")
        logger.info(f"Please add your PDF files to {config.pdf_dir}")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    try:
        processor = DataProcessor()
        vector_store = processor.process_all()
        logger.info("✅ PDF processing completed!")
        return True
    except Exception as e:
        logger.error(f"❌ PDF processing failed: {e}")
        return False

def start_api():
    """Start the FastAPI backend"""
    logger.info("🌐 Starting API server...")
    
    try:
        import uvicorn
        uvicorn.run(
            "api:app",
            host=config.api_host,
            port=config.api_port,
            reload=True,
            log_level=config.log_level.lower()
        )
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")

def start_ui():
    """Start the Streamlit UI"""
    logger.info("🖥️  Starting Streamlit UI...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_ui.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except Exception as e:
        logger.error(f"❌ Failed to start UI: {e}")

def run_evaluation():
    """Run chatbot evaluation"""
    logger.info("📊 Running evaluation...")
    
    try:
        from evaluation import ChatbotEvaluator
        evaluator = ChatbotEvaluator()
        report = evaluator.run_evaluation()
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        summary = report.get("summary", {})
        print(f"Total test cases: {report.get('total_test_cases', 0)}")
        print(f"Successful evaluations: {report.get('successful_evaluations', 0)}")
        
        for metric, stats in summary.items():
            if metric != "category_performance" and isinstance(stats, dict):
                print(f"{metric.title()}: {stats.get('mean', 0):.2f}")
        
        logger.info("✅ Evaluation completed!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")

def test_system():
    """Test system components"""
    logger.info("🧪 Testing system components...")
    
    # Test configuration
    try:
        from config import config
        logger.info("✅ Configuration loaded")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False
    
    # Test database
    try:
        from database import get_session
        db = next(get_session())
        db.close()
        logger.info("✅ Database connection OK")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False
    
    # Test chatbot
    try:
        from chatbot import get_chatbot
        chatbot = get_chatbot()
        
        # Test basic functionality
        session_id = chatbot.create_session("test_user")
        result = chatbot.chat("What is AI?", session_id)
        
        logger.info("✅ Chatbot basic functionality OK")
        logger.info(f"   Tool used: {result['tool_used']}")
        logger.info(f"   Response length: {len(result['response'])} chars")
        
    except Exception as e:
        logger.error(f"❌ Chatbot error: {e}")
        return False
    
    # Test tools
    try:
        from tools import WebSearchTool, MathTool, QueryRouter
        
        # Test math tool
        math_tool = MathTool()
        result = math_tool.calculate("2 + 2")
        logger.info("✅ Math tool OK")
        
        # Test web search (if configured)
        web_tool = WebSearchTool()
        if web_tool.serp_api or web_tool.tavily_client:
            logger.info("✅ Web search configured")
        else:
            logger.warning("⚠️  Web search not configured (no API keys)")
        
        # Test router
        router = QueryRouter()
        route = router.route("What is 2+2?")
        logger.info(f"✅ Query router OK (test route: {route})")
        
    except Exception as e:
        logger.error(f"❌ Tools error: {e}")
        return False
    
    logger.info("✅ All system tests passed!")
    return True

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Context-Aware Research Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup           # Initial setup
  python main.py process-pdfs    # Process PDF files
  python main.py start-api       # Start API server
  python main.py start-ui        # Start Streamlit UI
  python main.py test            # Test system
  python main.py eval            # Run evaluation
        """
    )
    
    parser.add_argument("command", choices=[
        "setup", "process-pdfs", "start-api", "start-ui", 
        "test", "eval", "all"
    ], help="Command to execute")
    
    parser.add_argument("--force", action="store_true", 
                       help="Force reprocessing")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_project()
        
    elif args.command == "process-pdfs":
        if not config.vector_store_dir.exists() or args.force:
            process_pdfs()
        else:
            logger.info("Vector store already exists. Use --force to reprocess.")
            
    elif args.command == "start-api":
        start_api()
        
    elif args.command == "start-ui":
        start_ui()
        
    elif args.command == "test":
        test_system()
        
    elif args.command == "eval":
        run_evaluation()
        
    elif args.command == "all":
        # Run complete workflow
        logger.info("🚀 Running complete workflow...")
        
        if not setup_project():
            return
        
        if not test_system():
            return
            
        # Check for PDFs and process if needed
        pdf_files = list(config.pdf_dir.glob("*.pdf"))
        if pdf_files and not config.vector_store_dir.exists():
            if not process_pdfs():
                logger.warning("⚠️  PDF processing failed, but continuing...")
        
        logger.info("✅ Setup complete!")
        logger.info("🌐 You can now:")
        logger.info(f"   - Start API: python main.py start-api")
        logger.info(f"   - Start UI: python main.py start-ui")
        logger.info(f"   - Run tests: python main.py test")
        logger.info(f"   - Run evaluation: python main.py eval")

if __name__ == "__main__":
    main()