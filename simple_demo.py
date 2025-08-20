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
                    
                    # Initialize components with latest GPT-4o model
                    llm = ChatOpenAI(
                        model="gpt-4o",  # Latest GPT-4 Omni model (most advanced available)
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
                    
                    # Search for relevant documents (increased from 5 to 10 for more comprehensive results)
                    docs = vector_store.similarity_search(prompt, k=10)
                    
                    # Create detailed context from retrieved documents
                    context_parts = []
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        context_parts.append(f"Document {i} (Source: {source}, Page {page}):\n{doc.page_content}")
                    
                    context = "\n\n" + "="*80 + "\n\n".join(context_parts)
                    
                    # Generate response with enhanced prompt for detailed answers
                    response_prompt = f"""You are an expert AI policy researcher. Provide a comprehensive, detailed answer based on the extensive context provided from AI policy documents.

CONTEXT FROM AI POLICY DOCUMENTS:
{context}

QUESTION: {prompt}

INSTRUCTIONS:
1. Provide a thorough, detailed answer drawing from ALL relevant parts of the context
2. Include specific information, examples, frameworks, and recommendations from the documents  
3. Quote important passages directly when they provide key insights
4. Organize your response with clear sections and subsections if the topic is complex
5. Explain technical concepts in detail based on the document content
6. Include specific policy recommendations, guidelines, or frameworks mentioned in the documents
7. If multiple perspectives or approaches are mentioned, discuss them all
8. Provide concrete examples from the documents when available
9. Make your response comprehensive - aim for 400-800 words when the topic warrants it
10. At the end, cite which specific documents and pages contained the most relevant information

COMPREHENSIVE DETAILED ANSWER:"""
                    
                    response = llm.invoke(response_prompt).content
                    
                    st.markdown(response)
                    
                    # Show detailed sources with content preview
                    if docs:
                        st.markdown("### 📚 Detailed Sources:")
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            
                            with st.expander(f"**Source {i}:** {source} (Page {page})"):
                                st.markdown(f"**Preview:** {preview}")
                                st.markdown(f"**Full Content:**")
                                st.text_area(f"Content from page {page}", doc.page_content, height=150, key=f"source_{i}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### 🔍 Sample Detailed Questions")
        sample_questions = [
            "What are the key AI safety guidelines and their implementation strategies?",
            "How does GDPR apply to AI systems and what are the compliance requirements?",
            "What are the comprehensive ethical considerations for AI deployment?", 
            "What specific regulations exist for AI in healthcare and their requirements?",
            "How should AI bias be addressed with detailed methodologies and frameworks?",
            "What are the complete AI governance frameworks mentioned in policy documents?",
            "Provide detailed information about AI transparency and explainability requirements"
        ]
        
        for q in sample_questions:
            if st.button(q):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        
        st.markdown("### 📊 Enhanced System Info")
        st.success(f"""
        - **Dataset**: AI Policy Documents
        - **Pages**: 1,207
        - **Chunks**: 4,808
        - **Retrieved per query**: 10 documents
        - **Model**: GPT-4o (Latest OpenAI Model) 🚀
        - **Embedding**: all-MiniLM-L6-v2
        - **Response style**: Comprehensive & Detailed
        - **Source preview**: Full document content
        """)

if __name__ == "__main__":
    main()
