"""
Demo script for the Context-Aware Research Chatbot
This script demonstrates various features and capabilities of the chatbot.
"""
import os
import time
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Import chatbot components
from chatbot import get_chatbot
from database import DatabaseManager, init_database
from evaluation import ChatbotEvaluator
from config import config

console = Console()

def display_header():
    """Display demo header"""
    header_text = """
# 🤖 Context-Aware Research Chatbot Demo

Welcome to the interactive demonstration of the Context-Aware Research Chatbot!
This demo will showcase the chatbot's capabilities across different query types.
    """
    console.print(Panel(Markdown(header_text), title="Demo", border_style="blue"))

def display_system_info():
    """Display system configuration and status"""
    console.print("\n[bold yellow]📋 System Configuration[/bold yellow]")
    
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value")
    
    config_table.add_row("LLM Model", config.llm_model)
    config_table.add_row("Embedding Model", config.embedding_model)
    config_table.add_row("Vector Store Type", config.vector_store_type)
    config_table.add_row("Chunk Size", str(config.chunk_size))
    config_table.add_row("Top K Retrieval", str(config.top_k_retrieval))
    config_table.add_row("Memory Window", str(config.memory_window))
    
    console.print(config_table)

def check_system_health():
    """Check system components health"""
    console.print("\n[bold yellow]🏥 System Health Check[/bold yellow]")
    
    health_table = Table(show_header=True, header_style="bold magenta")
    health_table.add_column("Component", style="dim")
    health_table.add_column("Status")
    health_table.add_column("Details")
    
    # Check database
    try:
        db_manager = DatabaseManager()
        stats = db_manager.get_global_stats()
        health_table.add_row("Database", "[green]✅ Healthy[/green]", f"Sessions: {stats['total_sessions']}")
    except Exception as e:
        health_table.add_row("Database", "[red]❌ Error[/red]", str(e))
    
    # Check chatbot
    try:
        chatbot = get_chatbot()
        health_table.add_row("Chatbot", "[green]✅ Healthy[/green]", "Initialized successfully")
    except Exception as e:
        health_table.add_row("Chatbot", "[red]❌ Error[/red]", str(e))
    
    # Check vector store
    try:
        if config.vector_store_dir.exists():
            health_table.add_row("Vector Store", "[green]✅ Available[/green]", "RAG enabled")
        else:
            health_table.add_row("Vector Store", "[yellow]⚠️ Missing[/yellow]", "RAG disabled")
    except Exception as e:
        health_table.add_row("Vector Store", "[red]❌ Error[/red]", str(e))
    
    # Check API keys
    api_status = "Configured" if config.openai_api_key else "Missing"
    api_color = "green" if config.openai_api_key else "red"
    health_table.add_row("OpenAI API", f"[{api_color}]{api_status}[/{api_color}]", "Required for LLM")
    
    web_search_status = "Available" if (config.serpapi_key or config.tavily_api_key) else "Not configured"
    web_search_color = "green" if (config.serpapi_key or config.tavily_api_key) else "yellow"
    health_table.add_row("Web Search", f"[{web_search_color}]{web_search_status}[/{web_search_color}]", "Optional feature")
    
    console.print(health_table)

def demo_query_routing():
    """Demonstrate query routing capabilities"""
    console.print("\n[bold yellow]🧠 Query Routing Demo[/bold yellow]")
    
    chatbot = get_chatbot()
    session_id = chatbot.create_session("demo_user")
    
    # Test queries for different tools
    test_queries = [
        {
            "query": "What is GDPR and how does it apply to AI systems?",
            "expected_tool": "rag",
            "description": "Knowledge base query about AI policy"
        },
        {
            "query": "Calculate the compound interest on $10,000 at 5% annually for 10 years",
            "expected_tool": "math", 
            "description": "Mathematical calculation"
        },
        {
            "query": "What are the latest developments in AI safety research?",
            "expected_tool": "web_search",
            "description": "Current events and recent information"
        }
    ]
    
    routing_table = Table(show_header=True, header_style="bold magenta")
    routing_table.add_column("Query", style="dim", width=40)
    routing_table.add_column("Expected Tool", justify="center")
    routing_table.add_column("Actual Tool", justify="center") 
    routing_table.add_column("Status", justify="center")
    
    for test_case in test_queries:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Processing: {test_case['query'][:30]}...", total=1)
            
            try:
                result = chatbot.chat(test_case["query"], session_id)
                actual_tool = result["tool_used"]
                expected_tool = test_case["expected_tool"]
                
                status = "[green]✅ Correct[/green]" if actual_tool == expected_tool else "[yellow]⚠️ Different[/yellow]"
                
                routing_table.add_row(
                    test_case["query"][:40] + "..." if len(test_case["query"]) > 40 else test_case["query"],
                    expected_tool,
                    actual_tool,
                    status
                )
                
                progress.update(task, advance=1)
                time.sleep(0.5)  # Brief pause for demo effect
                
            except Exception as e:
                routing_table.add_row(
                    test_case["query"][:40] + "..." if len(test_case["query"]) > 40 else test_case["query"],
                    expected_tool,
                    "[red]Error[/red]",
                    "[red]❌ Failed[/red]"
                )
    
    console.print(routing_table)

def demo_interactive_chat():
    """Interactive chat demonstration"""
    console.print("\n[bold yellow]💬 Interactive Chat Demo[/bold yellow]")
    console.print("Type your questions below. Type 'quit' to exit this demo section.\n")
    
    chatbot = get_chatbot()
    session_id = chatbot.create_session("interactive_demo_user")
    
    while True:
        try:
            # Get user input
            user_input = console.input("[bold blue]You:[/bold blue] ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("[dim]Exiting interactive demo...[/dim]")
                break
            
            if not user_input.strip():
                continue
            
            # Process query
            with Progress(
                SpinnerColumn(),
                TextColumn("🤖 Thinking..."),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing query...", total=1)
                
                result = chatbot.chat(user_input, session_id)
                
                progress.update(task, advance=1)
            
            # Display response
            console.print(f"\n[bold green]🤖 Bot:[/bold green] {result['response']}")
            
            # Display metadata
            metadata_table = Table(show_header=False, box=None, padding=(0, 1))
            metadata_table.add_column("", style="dim")
            metadata_table.add_column("", style="dim")
            
            metadata_table.add_row("🔧 Tool Used:", result['tool_used'])
            metadata_table.add_row("📚 Sources:", str(len(result['sources'])))
            
            console.print(metadata_table)
            
            # Display sources if available
            if result['sources']:
                console.print("\n[dim]Sources:[/dim]")
                for i, source in enumerate(result['sources'], 1):
                    if 'file' in source:
                        console.print(f"[dim]  {i}. {source['file']} (Page {source.get('page', 'N/A')})[/dim]")
                    elif 'source' in source:
                        console.print(f"[dim]  {i}. {source['source']}[/dim]")
            
            console.print()  # Empty line for spacing
            
        except KeyboardInterrupt:
            console.print("\n[dim]Demo interrupted by user[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def demo_evaluation():
    """Demonstrate evaluation capabilities"""
    console.print("\n[bold yellow]📊 Evaluation Demo[/bold yellow]")
    
    try:
        evaluator = ChatbotEvaluator()
        
        # Create small test dataset
        test_dataset = [
            {
                "question": "What is 15% of 200?",
                "expected_answer": "30",
                "expected_tool": "math",
                "category": "calculation"
            },
            {
                "question": "What are AI ethics principles?",
                "expected_answer": "AI ethics principles include fairness, transparency, accountability, and privacy.",
                "expected_tool": "rag",
                "category": "knowledge"
            }
        ]
        
        console.print("Running evaluation on sample questions...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("Evaluating chatbot performance...", total=len(test_dataset))
            
            report = evaluator.run_evaluation(test_dataset)
            
            progress.update(task, advance=len(test_dataset))
        
        # Display results
        console.print("\n[bold green]Evaluation Results:[/bold green]")
        
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="dim")
        results_table.add_column("Score", justify="center")
        results_table.add_column("Count", justify="center")
        
        summary = report.get('summary', {})
        for metric, stats in summary.items():
            if metric != "category_performance" and isinstance(stats, dict):
                score = f"{stats.get('mean', 0):.2f}"
                count = str(stats.get('count', 0))
                results_table.add_row(metric.title(), score, count)
        
        console.print(results_table)
        
        # Display category performance
        category_perf = summary.get('category_performance', {})
        if category_perf:
            console.print("\n[bold green]Category Performance:[/bold green]")
            for category, stats in category_perf.items():
                accuracy = stats.get('accuracy', 0) * 100
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                console.print(f"  {category}: {accuracy:.1f}% ({correct}/{total})")
    
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        console.print("[dim]This might be due to missing API keys or other configuration issues.[/dim]")

def demo_session_analytics():
    """Demonstrate session analytics"""
    console.print("\n[bold yellow]📈 Session Analytics Demo[/bold yellow]")
    
    try:
        db_manager = DatabaseManager()
        
        # Get global stats
        global_stats = db_manager.get_global_stats()
        
        analytics_table = Table(show_header=True, header_style="bold magenta")
        analytics_table.add_column("Metric", style="dim")
        analytics_table.add_column("Value", justify="center")
        
        analytics_table.add_row("Total Sessions", str(global_stats.get('total_sessions', 0)))
        analytics_table.add_row("Total Messages", str(global_stats.get('total_messages', 0)))
        analytics_table.add_row("Recent Active Sessions", str(global_stats.get('recent_active_sessions', 0)))
        
        console.print(analytics_table)
        
        # Tool usage stats
        tool_usage = global_stats.get('tool_usage', {})
        if tool_usage:
            console.print("\n[bold green]Tool Usage Statistics:[/bold green]")
            for tool, count in tool_usage.items():
                console.print(f"  🔧 {tool.replace('_', ' ').title()}: {count}")
        else:
            console.print("\n[dim]No tool usage data available yet.[/dim]")
    
    except Exception as e:
        console.print(f"[red]Analytics failed: {e}[/red]")

def main():
    """Main demo function"""
    try:
        # Initialize database
        init_database()
        
        # Display header
        display_header()
        
        # System information
        display_system_info()
        check_system_health()
        
        # Demo sections
        console.print("\n" + "="*70)
        demo_query_routing()
        
        console.print("\n" + "="*70)
        demo_session_analytics()
        
        console.print("\n" + "="*70)
        demo_evaluation()
        
        console.print("\n" + "="*70)
        demo_interactive_chat()
        
        # Final message
        console.print("\n[bold green]🎉 Demo completed![/bold green]")
        console.print("\n[dim]To continue exploring:")
        console.print("  • Start the API: python main.py start-api")
        console.print("  • Launch Streamlit UI: python main.py start-ui")
        console.print("  • Launch Gradio UI: python gradio_ui.py")
        console.print("  • Run evaluation: python main.py eval")
        console.print("  • Process new PDFs: python main.py process-pdfs[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo failed: {e}[/red]")
        console.print("[dim]Please check your configuration and try again.[/dim]")

if __name__ == "__main__":
    main()