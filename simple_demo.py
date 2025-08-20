#!/usr/bin/env python3
"""
Simple demo of the AI Policy Chatbot without complex dependencies
"""
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Simple configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(
        page_title="AI Policy Research Chatbot", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Policy Research Chatbot")
    st.markdown("### Ask questions about AI policy and get cited responses!")
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        st.error("❌ Please add your OpenAI API key to the .env file")
        st.info("Edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key")
        return
    
    # Check if vector store exists
    vector_store_path = Path("data/vector_store")
    if not vector_store_path.exists():
        st.error("❌ Vector store not found. Please run: `python main.py process-pdfs --force`")
        return
    
    st.success("✅ AI Policy dataset loaded (4,808 chunks from 1,207 pages)")
    
    # Simple chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about AI policy..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching AI policy documents..."):
                try:
                    # Import here to avoid early errors
                    import os
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
                    
                    from langchain_openai import ChatOpenAI
                    from langchain_community.vectorstores import FAISS
                    from langchain_huggingface import HuggingFaceEmbeddings
                    
                    # Initialize components
                    llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.0,
                        openai_api_key=OPENAI_API_KEY
                    )
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Load vector store
                    vector_store = FAISS.load_local(
                        str(vector_store_path), 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Search for relevant documents
                    docs = vector_store.similarity_search(prompt, k=5)
                    
                    # Create context from retrieved documents
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate response
                    response_prompt = f"""You are an AI policy expert. Answer the user's question based on the provided context.
                    
Context from AI policy documents:
{context}

Question: {prompt}

Instructions:
1. Provide a comprehensive answer based on the context
2. If the context doesn't contain enough information, say so
3. Always cite your sources using [Source: filename, page/section]
4. Be precise and factual

Answer:"""
                    
                    response = llm.invoke(response_prompt).content
                    
                    st.markdown(response)
                    
                    # Show sources
                    if docs:
                        st.markdown("### 📚 Sources:")
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            st.markdown(f"**{i}.** {source} (Page {page})")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### 🔍 Sample Questions")
        sample_questions = [
            "What are the key AI safety guidelines?",
            "How does GDPR apply to AI systems?",
            "What are the ethical considerations for AI?",
            "What regulations exist for AI in healthcare?",
            "How should AI bias be addressed?"
        ]
        
        for q in sample_questions:
            if st.button(q):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        
        st.markdown("### 📊 System Info")
        st.info(f"""
        - **Dataset**: AI Policy Documents
        - **Pages**: 1,207
        - **Chunks**: 4,808
        - **Model**: GPT-3.5-turbo
        - **Embedding**: all-MiniLM-L6-v2
        """)

if __name__ == "__main__":
    main()
