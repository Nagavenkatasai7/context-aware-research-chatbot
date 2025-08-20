"""
Gradio UI for the Context-Aware Research Chatbot
"""
import gradio as gr
import requests
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Global state
current_session_id = None
current_user_id = "gradio_user"

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API calls to the backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def format_sources(sources: List[Dict]) -> str:
    """Format sources for display"""
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        if "file" in source:
            formatted.append(f"📄 **{source['file']}** (Page {source.get('page', 'N/A')})")
        elif "source" in source:
            if source['source'] == "web_search":
                formatted.append(f"🌐 **Web Search**")
            elif source['source'] == "calculator":
                formatted.append(f"🧮 **Calculator**")
            else:
                formatted.append(f"📚 **{source['source']}**")
        else:
            formatted.append(f"📋 Source {i}")
    
    return "\n".join(formatted)

def get_tool_emoji(tool: str) -> str:
    """Get emoji for tool type"""
    tool_emojis = {
        "rag": "📚",
        "web_search": "🌐", 
        "math": "🧮"
    }
    return tool_emojis.get(tool, "🔧")

def create_session():
    """Create a new chat session"""
    global current_session_id
    
    response = call_api("/sessions", "POST", {"user_id": current_user_id})
    if "session_id" in response:
        current_session_id = response["session_id"]
        return f"✅ New session created: {current_session_id[:8]}..."
    else:
        return f"❌ Failed to create session: {response.get('error', 'Unknown error')}"

def chat_fn(message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """Main chat function"""
    global current_session_id
    
    if not message.strip():
        return history, ""
    
    # Create session if needed
    if not current_session_id:
        session_response = call_api("/sessions", "POST", {"user_id": current_user_id})
        if "session_id" in session_response:
            current_session_id = session_response["session_id"]
        else:
            error_msg = f"❌ Failed to create session: {session_response.get('error', 'Unknown error')}"
            history.append((message, error_msg))
            return history, ""
    
    # Send message to chatbot
    chat_data = {
        "message": message,
        "session_id": current_session_id,
        "user_id": current_user_id
    }
    
    response = call_api("/chat", "POST", chat_data)
    
    if "error" not in response:
        # Format response with metadata
        tool_emoji = get_tool_emoji(response.get("tool_used", "unknown"))
        tool_name = response.get("tool_used", "unknown").replace("_", " ").title()
        
        formatted_response = response["response"]
        
        # Add tool information
        formatted_response += f"\n\n---\n**{tool_emoji} Tool Used:** {tool_name}"
        
        # Add sources if available
        sources = response.get("sources", [])
        if sources:
            sources_text = format_sources(sources)
            formatted_response += f"\n\n**📚 Sources:**\n{sources_text}"
        
        # Add routing explanation
        routing_explanation = response.get("routing_explanation", "")
        if routing_explanation:
            formatted_response += f"\n\n**🧠 Routing Logic:** {routing_explanation}"
        
        history.append((message, formatted_response))
    else:
        error_msg = f"❌ Error: {response['error']}"
        history.append((message, error_msg))
    
    return history, ""

def clear_chat():
    """Clear chat history"""
    global current_session_id
    
    if current_session_id:
        call_api(f"/sessions/{current_session_id}/clear")
    
    return [], "Chat cleared!"

def get_session_stats():
    """Get current session statistics"""
    global current_session_id
    
    if not current_session_id:
        return "No active session"
    
    stats_response = call_api(f"/sessions/{current_session_id}/stats")
    
    if "error" not in stats_response:
        stats_text = f"**Session ID:** {current_session_id[:8]}...\n"
        stats_text += f"**User ID:** {stats_response.get('user_id', 'Unknown')}\n"
        stats_text += f"**Total Messages:** {stats_response.get('total_messages', 0)}\n"
        
        tools_used = stats_response.get('tools_used', {})
        if tools_used:
            stats_text += "\n**Tools Used:**\n"
            for tool, count in tools_used.items():
                emoji = get_tool_emoji(tool)
                tool_name = tool.replace("_", " ").title()
                stats_text += f"• {emoji} {tool_name}: {count}\n"
        
        return stats_text
    else:
        return f"❌ Error getting stats: {stats_response['error']}"

def get_system_health():
    """Get system health status"""
    health_response = call_api("/health")
    
    if "error" not in health_response:
        status = health_response.get("status", "unknown")
        health_text = f"**System Status:** {'✅ Healthy' if status == 'healthy' else '❌ Issues'}\n"
        health_text += f"**RAG Available:** {'✅ Yes' if health_response.get('rag_available') else '❌ No'}\n"
        health_text += f"**Web Search:** {'✅ Yes' if health_response.get('web_search_available') else '❌ No'}\n"
        health_text += f"**Timestamp:** {health_response.get('timestamp', 'Unknown')}"
        
        return health_text
    else:
        return f"❌ Cannot connect to API: {health_response['error']}"

def get_global_stats():
    """Get global system statistics"""
    stats_response = call_api("/stats")
    
    if "error" not in stats_response:
        stats_text = f"**Total Sessions:** {stats_response.get('total_sessions', 0)}\n"
        stats_text += f"**Total Messages:** {stats_response.get('total_messages', 0)}\n"
        stats_text += f"**Recent Active Sessions:** {stats_response.get('recent_active_sessions', 0)}\n"
        
        tool_usage = stats_response.get('tool_usage', {})
        if tool_usage:
            stats_text += "\n**Global Tool Usage:**\n"
            for tool, count in tool_usage.items():
                emoji = get_tool_emoji(tool)
                tool_name = tool.replace("_", " ").title()
                stats_text += f"• {emoji} {tool_name}: {count}\n"
        
        return stats_text
    else:
        return f"❌ Error getting global stats: {stats_response['error']}"

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .tool-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    .sources-section {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .stats-container {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=custom_css,
        title="Context-Aware Research Chatbot"
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 Context-Aware Research Chatbot
            
            Ask questions about AI policy, get current information, or perform calculations!
            The system will automatically route your query to the most appropriate tool.
            """
        )
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            [],
                            elem_id="chatbot",
                            bubble_full_width=False,
                            height=500,
                            show_label=False
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Ask me anything about AI policy, current events, or math...",
                                show_label=False,
                                scale=4
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                            new_session_btn = gr.Button("🆕 New Session", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Session Info")
                        session_stats = gr.Markdown("No active session")
                        refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                        
                        gr.Markdown("### 🏥 System Health")
                        system_health = gr.Markdown("Checking...")
                        refresh_health_btn = gr.Button("🔄 Check Health")
                
                # Event handlers
                def submit_message(message, history):
                    return chat_fn(message, history)
                
                def clear_and_refresh():
                    history, status = clear_chat(), []
                    stats = get_session_stats()
                    return history, stats
                
                def new_session_and_refresh():
                    status = create_session()
                    stats = get_session_stats()
                    return [], stats, status
                
                # Bind events
                submit_btn.click(
                    submit_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg]
                )
                
                msg.submit(
                    submit_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg]
                )
                
                clear_btn.click(
                    clear_and_refresh,
                    outputs=[chatbot, session_stats]
                )
                
                new_session_btn.click(
                    new_session_and_refresh,
                    outputs=[chatbot, session_stats, gr.Textbox(visible=False)]
                )
                
                refresh_stats_btn.click(
                    get_session_stats,
                    outputs=session_stats
                )
                
                refresh_health_btn.click(
                    get_system_health,
                    outputs=system_health
                )
            
            # Analytics Tab
            with gr.Tab("📈 Analytics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 Global Statistics")
                        global_stats = gr.Markdown("Loading...")
                        refresh_global_btn = gr.Button("🔄 Refresh Global Stats")
                        
                        refresh_global_btn.click(
                            get_global_stats,
                            outputs=global_stats
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🏥 System Status")
                        system_status = gr.Markdown("Loading...")
                        refresh_system_btn = gr.Button("🔄 Refresh System Status")
                        
                        refresh_system_btn.click(
                            get_system_health,
                            outputs=system_status
                        )
            
            # Help Tab
            with gr.Tab("❓ Help"):
                gr.Markdown(
                    """
                    ## 🎯 Query Types & Examples
                    
                    The chatbot automatically routes your queries to the most appropriate tool:
                    
                    ### 📚 Knowledge Base (RAG)
                    - "What is GDPR?"
                    - "AI Act requirements for high-risk systems"
                    - "Privacy regulations for AI"
                    
                    ### 🌐 Web Search
                    - "Latest AI news today"
                    - "Recent AI safety developments"
                    - "Current AI policy updates"
                    
                    ### 🧮 Math Calculator
                    - "Calculate 15% of 250000"
                    - "What is 2^10?"
                    - "45 * 123 + 678"
                    
                    ## 🔧 Features
                    
                    - **Smart Routing**: Queries are automatically routed to the best tool
                    - **Source Citations**: All responses include source attribution
                    - **Session Memory**: Conversations maintain context
                    - **Multiple Tools**: Web search, local knowledge, and calculations
                    
                    ## 💡 Tips
                    
                    - Be specific in your questions for better results
                    - Use keywords like "latest" or "recent" for web search
                    - Mathematical expressions are automatically detected
                    - Check the "Tool Used" section to see how your query was handled
                    
                    ## 🔗 API Endpoints
                    
                    If you're interested in the API:
                    - **API Docs**: http://localhost:8000/docs
                    - **Health Check**: http://localhost:8000/health
                    - **Stats**: http://localhost:8000/stats
                    """
                )
        
        # Initialize components on load
        demo.load(
            fn=lambda: [get_session_stats(), get_system_health(), get_global_stats()],
            outputs=[session_stats, system_health, global_stats]
        )
    
    return demo

def main():
    """Main function to run the Gradio interface"""
    
    # Check API availability
    try:
        health_response = call_api("/health")
        if "error" in health_response:
            print("❌ Cannot connect to API. Please make sure the backend is running on http://localhost:8000")
            return
    except:
        print("❌ Cannot connect to API. Please make sure the backend is running on http://localhost:8000")
        return
    
    # Create and launch interface
    demo = create_interface()
    
    print("🚀 Starting Gradio interface...")
    print("🌐 The interface will be available at: http://localhost:7860")
    print("📋 Make sure the API server is running at: http://localhost:8000")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        debug=False
    )

if __name__ == "__main__":
    main()