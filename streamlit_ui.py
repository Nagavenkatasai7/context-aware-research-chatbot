"""
Streamlit UI for the Context-Aware Research Chatbot
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Context-Aware Research Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.bot-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}
.source-item {
    background-color: #f5f5f5;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 0.25rem;
    font-size: 0.9rem;
}
.tool-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.tool-rag {
    background-color: #e8f5e8;
    color: #2e7d32;
}
.tool-web {
    background-color: #e3f2fd;
    color: #1976d2;
}
.tool-math {
    background-color: #fff3e0;
    color: #f57c00;
}
</style>
""", unsafe_allow_html=True)

# Utility functions
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
        st.error(f"API Error: {e}")
        return {"error": str(e)}

def get_tool_badge_class(tool: str) -> str:
    """Get CSS class for tool badge"""
    tool_classes = {
        "rag": "tool-rag",
        "web_search": "tool-web", 
        "math": "tool-math"
    }
    return tool_classes.get(tool, "tool-rag")

def format_sources(sources: List[Dict]) -> str:
    """Format sources for display"""
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        if "file" in source:
            formatted.append(f"{i}. **{source['file']}** (Page {source.get('page', 'N/A')})")
        elif "source" in source:
            formatted.append(f"{i}. **{source['source']}**")
        else:
            formatted.append(f"{i}. Source {i}")
    
    return "\n".join(formatted)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"

# Main app
def main():
    st.title("🤖 Context-Aware Research Chatbot")
    st.markdown("Ask questions about AI policy, get current information, or perform calculations!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # User ID input
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        if user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
        
        # Session management
        st.subheader("Session Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Session"):
                # Create new session
                response = call_api("/sessions", "POST", {"user_id": st.session_state.user_id})
                if "session_id" in response:
                    st.session_state.session_id = response["session_id"]
                    st.session_state.messages = []
                    st.success("New session created!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                if st.session_state.session_id:
                    call_api(f"/sessions/{st.session_state.session_id}/clear")
                st.session_state.messages = []
                st.success("Chat cleared!")
                st.rerun()
        
        # Display current session
        if st.session_state.session_id:
            st.info(f"**Session:** {st.session_state.session_id[:8]}...")
        
        # Session stats
        if st.session_state.session_id:
            st.subheader("📊 Session Stats")
            stats_response = call_api(f"/sessions/{st.session_state.session_id}/stats")
            if "error" not in stats_response:
                st.metric("Messages", stats_response.get("total_messages", 0))
                tools_used = stats_response.get("tools_used", {})
                if tools_used:
                    st.write("**Tools Used:**")
                    for tool, count in tools_used.items():
                        st.write(f"• {tool}: {count}")
        
        # System status
        st.subheader("🏥 System Status")
        health_response = call_api("/health")
        if "error" not in health_response:
            status = health_response.get("status", "unknown")
            if status == "healthy":
                st.success("✅ System Healthy")
            else:
                st.error("❌ System Issues")
            
            st.write(f"**RAG Available:** {'✅' if health_response.get('rag_available') else '❌'}")
            st.write(f"**Web Search:** {'✅' if health_response.get('web_search_available') else '❌'}")
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    # Message content
                    st.write(message["content"])
                    
                    # Tool badge
                    tool_used = message.get("tool_used", "unknown")
                    tool_class = get_tool_badge_class(tool_used)
                    st.markdown(f'<span class="tool-badge {tool_class}">🔧 {tool_used.replace("_", " ").title()}</span>', 
                              unsafe_allow_html=True)
                    
                    # Sources
                    if message.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            sources_text = format_sources(message["sources"])
                            st.markdown(sources_text)
                    
                    # Routing explanation
                    if message.get("routing_explanation"):
                        with st.expander("🧠 Routing Logic", expanded=False):
                            st.write(message["routing_explanation"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about AI policy, current events, or math..."):
        # Create session if needed
        if not st.session_state.session_id:
            response = call_api("/sessions", "POST", {"user_id": st.session_state.user_id})
            if "session_id" in response:
                st.session_state.session_id = response["session_id"]
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_data = {
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id
                }
                
                response = call_api("/chat", "POST", chat_data)
                
                if "error" not in response:
                    # Display response
                    st.write(response["response"])
                    
                    # Tool badge
                    tool_used = response.get("tool_used", "unknown")
                    tool_class = get_tool_badge_class(tool_used)
                    st.markdown(f'<span class="tool-badge {tool_class}">🔧 {tool_used.replace("_", " ").title()}</span>', 
                              unsafe_allow_html=True)
                    
                    # Sources
                    if response.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            sources_text = format_sources(response["sources"])
                            st.markdown(sources_text)
                    
                    # Routing explanation
                    if response.get("routing_explanation"):
                        with st.expander("🧠 Routing Logic", expanded=False):
                            st.write(response["routing_explanation"])
                    
                    # Add to message history
                    bot_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "tool_used": response.get("tool_used"),
                        "sources": response.get("sources", []),
                        "routing_explanation": response.get("routing_explanation")
                    }
                    st.session_state.messages.append(bot_message)
                else:
                    st.error(f"Error: {response['error']}")

# Admin page
def admin_page():
    st.title("🔧 Admin Dashboard")
    
    # Global stats
    st.header("📊 Global Statistics")
    stats_response = call_api("/stats")
    if "error" not in stats_response:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", stats_response.get("total_sessions", 0))
        with col2:
            st.metric("Total Messages", stats_response.get("total_messages", 0))
        with col3:
            st.metric("Recent Active", stats_response.get("recent_active_sessions", 0))
        with col4:
            avg_messages = (stats_response.get("total_messages", 0) / 
                          max(stats_response.get("total_sessions", 1), 1))
            st.metric("Avg Messages/Session", f"{avg_messages:.1f}")
        
        # Tool usage chart
        st.subheader("🔧 Tool Usage")
        tool_usage = stats_response.get("tool_usage", {})
        if tool_usage:
            df = pd.DataFrame(list(tool_usage.items()), columns=["Tool", "Usage Count"])
            st.bar_chart(df.set_index("Tool"))
    
    # System health
    st.header("🏥 System Health")
    health_response = call_api("/health")
    if "error" not in health_response:
        status = health_response.get("status", "unknown")
        if status == "healthy":
            st.success("✅ System is healthy")
        else:
            st.error("❌ System has issues")
        
        # Component status
        components = {
            "RAG System": health_response.get("rag_available", False),
            "Web Search": health_response.get("web_search_available", False)
        }
        
        for component, available in components.items():
            if available:
                st.success(f"✅ {component}: Available")
            else:
                st.error(f"❌ {component}: Not Available")
    
    # Configuration
    st.header("⚙️ Configuration")
    config_response = call_api("/config")
    if "error" not in config_response:
        st.json(config_response)
    
    # Cleanup
    st.header("🧹 Maintenance")
    st.subheader("Session Cleanup")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Delete sessions older than (days)", min_value=1, value=30)
    with col2:
        if st.button("🗑️ Start Cleanup"):
            response = call_api(f"/admin/cleanup?days={days}", "POST")
            if "error" not in response:
                st.success(response.get("message", "Cleanup started"))
            else:
                st.error(response["error"])

# Navigation
def main_app():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["💬 Chat", "🔧 Admin"])
    
    if page == "💬 Chat":
        main()
    elif page == "🔧 Admin":
        admin_page()

if __name__ == "__main__":
    # Check if API is available
    try:
        health_response = call_api("/health")
        if "error" in health_response:
            st.error("❌ Cannot connect to API. Please make sure the backend is running on http://localhost:8000")
            st.stop()
    except:
        st.error("❌ Cannot connect to API. Please make sure the backend is running on http://localhost:8000")
        st.stop()
    
    main_app()