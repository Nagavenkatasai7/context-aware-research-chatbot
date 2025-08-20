"""
Startup script for the Context-Aware Research Chatbot
Orchestrates the complete system startup process
"""
import os
import sys
import time
import signal
import subprocess
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table

# Local imports
from config import config
from database import init_database
from data_processor import DataProcessor

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages multiple services for the chatbot system"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.services = {
            "api": {
                "command": [sys.executable, "api.py"],
                "port": config.api_port,
                "health_url": f"http://localhost:{config.api_port}/health",
                "description": "FastAPI Backend"
            },
            "streamlit": {
                "command": [
                    sys.executable, "-m", "streamlit", "run", 
                    "streamlit_ui.py", 
                    "--server.port", "8501",
                    "--server.address", "0.0.0.0",
                    "--server.headless", "true"
                ],
                "port": 8501,
                "health_url": "http://localhost:8501/_stcore/health",
                "description": "Streamlit UI"
            },
            "gradio": {
                "command": [sys.executable, "gradio_ui.py"],
                "port": 7860,
                "health_url": "http://localhost:7860",
                "description": "Gradio UI"
            }
        }
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print(f"\n[yellow]Received signal {signum}, shutting down services...[/yellow]")
        self.shutdown_requested = True
        self.stop_all_services()
        sys.exit(0)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        console.print("[bold yellow]🔍 Checking Prerequisites[/bold yellow]")
        
        prerequisites = []
        
        # Check OpenAI API key
        if config.openai_api_key:
            prerequisites.append(("OpenAI API Key", "✅", "Configured"))
        else:
            prerequisites.append(("OpenAI API Key", "❌", "Missing - Required"))
            
        # Check database
        try:
            init_database()
            prerequisites.append(("Database", "✅", "Initialized"))
        except Exception as e:
            prerequisites.append(("Database", "❌", f"Error: {e}"))
        
        # Check vector store
        if config.vector_store_dir.exists():
            prerequisites.append(("Vector Store", "✅", "Available"))
        else:
            prerequisites.append(("Vector Store", "⚠️", "Missing - RAG disabled"))
        
        # Check PDF directory
        pdf_count = len(list(config.pdf_dir.glob("*.pdf")))
        if pdf_count > 0:
            prerequisites.append(("PDF Documents", "✅", f"{pdf_count} files found"))
        else:
            prerequisites.append(("PDF Documents", "⚠️", "No PDFs found"))
        
        # Display results
        prereq_table = Table(show_header=True, header_style="bold magenta")
        prereq_table.add_column("Component", style="dim")
        prereq_table.add_column("Status", justify="center")
        prereq_table.add_column("Details")
        
        for component, status, details in prerequisites:
            prereq_table.add_row(component, status, details)
        
        console.print(prereq_table)
        
        # Check if critical requirements are met
        critical_failed = any("❌" in status for _, status, _ in prerequisites 
                            if "API Key" in _ or "Database" in _)
        
        return not critical_failed
    
    def process_pdfs_if_needed(self) -> bool:
        """Process PDFs if vector store doesn't exist"""
        if config.vector_store_dir.exists():
            console.print("[dim]Vector store already exists, skipping PDF processing[/dim]")
            return True
        
        pdf_files = list(config.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            console.print("[yellow]No PDF files found, skipping processing[/yellow]")
            return True
        
        console.print(f"[bold yellow]📚 Processing {len(pdf_files)} PDF files[/bold yellow]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task("Processing PDFs and creating vector store...", total=1)
                
                processor = DataProcessor()
                processor.process_all()
                
                progress.update(task, advance=1)
            
            console.print("[green]✅ PDF processing completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ PDF processing failed: {e}[/red]")
            return False
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name in self.processes:
            console.print(f"[yellow]Service {service_name} already running[/yellow]")
            return True
        
        service_config = self.services[service_name]
        
        try:
            # Start the process
            process = subprocess.Popen(
                service_config["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes[service_name] = process
            
            # Wait for service to be ready
            max_retries = 30
            for i in range(max_retries):
                if self.shutdown_requested:
                    return False
                
                try:
                    response = requests.get(service_config["health_url"], timeout=2)
                    if response.status_code == 200:
                        console.print(f"[green]✅ {service_config['description']} started on port {service_config['port']}[/green]")
                        return True
                except:
                    pass
                
                time.sleep(2)
            
            console.print(f"[red]❌ {service_config['description']} failed to start[/red]")
            return False
            
        except Exception as e:
            console.print(f"[red]❌ Failed to start {service_name}: {e}[/red]")
            return False
    
    def stop_service(self, service_name: str):
        """Stop a specific service"""
        if service_name not in self.processes:
            return
        
        process = self.processes[service_name]
        service_config = self.services[service_name]
        
        try:
            process.terminate()
            process.wait(timeout=10)
            console.print(f"[dim]✅ {service_config['description']} stopped[/dim]")
        except subprocess.TimeoutExpired:
            process.kill()
            console.print(f"[dim]🔨 {service_config['description']} force killed[/dim]")
        except Exception as e:
            console.print(f"[red]Error stopping {service_name}: {e}[/red]")
        finally:
            del self.processes[service_name]
    
    def stop_all_services(self):
        """Stop all running services"""
        console.print("[yellow]🛑 Stopping all services...[/yellow]")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        console.print("[green]✅ All services stopped[/green]")
    
    def get_service_status(self) -> Table:
        """Get current status of all services"""
        status_table = Table(show_header=True, header_style="bold magenta")
        status_table.add_column("Service", style="dim")
        status_table.add_column("Status", justify="center")
        status_table.add_column("Port", justify="center")
        status_table.add_column("URL")
        
        for service_name, service_config in self.services.items():
            if service_name in self.processes:
                process = self.processes[service_name]
                if process.poll() is None:  # Process is running
                    try:
                        response = requests.get(service_config["health_url"], timeout=2)
                        if response.status_code == 200:
                            status = "[green]🟢 Running[/green]"
                        else:
                            status = "[yellow]🟡 Issues[/yellow]"
                    except:
                        status = "[yellow]🟡 No Response[/yellow]"
                else:
                    status = "[red]🔴 Crashed[/red]"
            else:
                status = "[red]🔴 Stopped[/red]"
            
            url = f"http://localhost:{service_config['port']}"
            status_table.add_row(
                service_config["description"],
                status,
                str(service_config["port"]),
                url
            )
        
        return status_table
    
    def start_all_services(self) -> bool:
        """Start all services in order"""
        console.print("[bold yellow]🚀 Starting Services[/bold yellow]")
        
        # Start API first (others depend on it)
        if not self.start_service("api"):
            return False
        
        # Start UIs
        success = True
        for service_name in ["streamlit", "gradio"]:
            if not self.start_service(service_name):
                success = False
        
        return success
    
    def monitor_services(self):
        """Monitor services and restart if needed"""
        console.print("[bold yellow]📊 Monitoring Services[/bold yellow]")
        console.print("[dim]Press Ctrl+C to stop all services[/dim]")
        
        try:
            with Live(self.get_service_status(), refresh_per_second=0.5, console=console) as live:
                while not self.shutdown_requested:
                    # Update the status table
                    live.update(self.get_service_status())
                    
                    # Check for crashed services and restart
                    for service_name in list(self.processes.keys()):
                        process = self.processes[service_name]
                        if process.poll() is not None:  # Process has terminated
                            console.print(f"[red]Service {service_name} crashed, restarting...[/red]")
                            self.stop_service(service_name)
                            self.start_service(service_name)
                    
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            self.shutdown_requested = True
        
        self.stop_all_services()

def display_startup_info():
    """Display startup information"""
    startup_text = """
# 🤖 Context-Aware Research Chatbot

Starting up the complete chatbot system with:
• FastAPI backend for core functionality
• Streamlit UI for interactive chat
• Gradio UI for alternative interface
• PDF processing and vector storage
• Evaluation and monitoring
    """
    console.print(Panel(startup_text, title="System Startup", border_style="blue"))

def display_access_urls():
    """Display access URLs for the services"""
    urls_table = Table(show_header=True, header_style="bold magenta")
    urls_table.add_column("Service", style="dim")
    urls_table.add_column("URL", style="bold blue")
    urls_table.add_column("Description")
    
    urls_table.add_row("API Documentation", "http://localhost:8000/docs", "FastAPI automatic docs")
    urls_table.add_row("API Health", "http://localhost:8000/health", "System health check")
    urls_table.add_row("Streamlit UI", "http://localhost:8501", "Main chat interface")
    urls_table.add_row("Gradio UI", "http://localhost:7860", "Alternative chat interface")
    
    console.print("\n[bold green]🌐 Access URLs[/bold green]")
    console.print(urls_table)

def main():
    """Main startup function"""
    try:
        display_startup_info()
        
        # Initialize service manager
        manager = ServiceManager()
        
        # Check prerequisites
        if not manager.check_prerequisites():
            console.print("[red]❌ Prerequisites not met. Please check your configuration.[/red]")
            return False
        
        # Process PDFs if needed
        if not manager.process_pdfs_if_needed():
            console.print("[yellow]⚠️ PDF processing failed, but continuing...[/yellow]")
        
        # Start all services
        if not manager.start_all_services():
            console.print("[red]❌ Failed to start some services[/red]")
            manager.stop_all_services()
            return False
        
        # Display access information
        display_access_urls()
        
        # Start monitoring
        manager.monitor_services()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Startup failed: {e}[/red]")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)