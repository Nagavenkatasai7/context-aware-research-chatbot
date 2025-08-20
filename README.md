# 🤖 Context-Aware Research Chatbot

A sophisticated conversational agent that answers domain questions about AI policy using web search, local RAG (Retrieval-Augmented Generation), and mathematical tools with comprehensive source citations.

## ✨ Features

- **🧠 Multi-Modal Intelligence**: Combines web search, local knowledge base, and mathematical calculations
- **🎯 Smart Routing**: Automatically routes queries to the most appropriate tool based on intent
- **💬 Conversational Memory**: Maintains context across conversations with session management
- **📚 Source Citations**: Provides detailed source attributions for all responses
- **📊 Comprehensive Evaluation**: Built-in evaluation framework for faithfulness and groundedness
- **🎨 Multiple Interfaces**: FastAPI backend, Streamlit UI, and Gradio interface
- **🏗️ Scalable Architecture**: Modular design using LangChain components

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- (Optional) SerpAPI or Tavily API key for web search
- PDF documents for your domain knowledge base

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/context-aware-research-chatbot.git
cd context-aware-research-chatbot
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Initialize the project:**
```bash
python main.py setup
```

5. **Add your PDF documents:**
```bash
# Place your PDF files in data/pdfs/
cp your-documents/*.pdf data/pdfs/
```

6. **Process documents:**
```bash
python main.py process-pdfs
```

7. **Test the system:**
```bash
python main.py test
```

## 🖥️ Usage

### Option 1: Streamlit UI (Recommended)
```bash
streamlit run simple_demo.py --server.port 8501
```
Access at: http://localhost:8501

### Option 2: FastAPI + Streamlit UI
```bash
# Terminal 1: Start API
python main.py start-api

# Terminal 2: Start UI
python main.py start-ui
```

### Option 3: Gradio Interface
```bash
python gradio_ui.py
```
Access at: http://localhost:7860

## 🎯 Sample Queries

Try these questions with your AI policy dataset:

- **Policy Questions**: "What are the key AI safety guidelines?"
- **Regulatory**: "How does GDPR apply to AI systems?"
- **Ethics**: "What are the ethical considerations for AI deployment?"
- **Math**: "Calculate 15% of 250,000"
- **Complex**: "How do AI policy frameworks address bias in algorithmic decision-making?"

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Gradio UI     │    │   FastAPI      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Context-Aware         │
                    │    Research Chatbot      │
                    │  ┌─────────────────────┐ │
                    │  │   Query Router      │ │
                    │  └─────────┬───────────┘ │
                    │  ┌─────────▼───────────┐ │
                    │  │      Tools          │ │
                    │  │ ┌─────┐ ┌─────┐     │ │
                    │  │ │ RAG │ │ Web │     │ │
                    │  │ └─────┘ └─────┘     │ │
                    │  │ ┌─────┐             │ │
                    │  │ │Math │             │ │
                    │  │ └─────┘             │ │
                    │  └─────────────────────┘ │
                    │  ┌─────────────────────┐ │
                    │  │   Memory Manager    │ │
                    │  └─────────────────────┘ │
                    └──────────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Data Layer          │
                    │ ┌─────┐ ┌─────┐ ┌─────┐  │
                    │ │FAISS│ │SQLite│ │PDFs │  │
                    │ └─────┘ └─────┘ └─────┘  │
                    └──────────────────────────┘
```

## 📁 Project Structure

```
context-aware-research-chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                 # Main CLI interface
├── config.py               # Configuration management
├── simple_demo.py          # Simplified Streamlit demo
├── data_processor.py       # PDF processing & vector store
├── tools.py                # Web search, math, RAG tools
├── chatbot.py              # Core chatbot logic
├── database.py             # Database models & management
├── api.py                  # FastAPI backend
├── streamlit_ui.py         # Streamlit frontend
├── gradio_ui.py            # Gradio frontend
├── evaluation.py           # Evaluation framework
└── data/                   # Data directory
    ├── pdfs/               # Place your PDF files here
    ├── vector_store/       # Generated vector store
    └── eval_dataset.json   # Evaluation dataset
```

## ⚙️ Configuration

Key configuration options in `.env`:

```env
# Required
OPENAI_API_KEY=your_key_here

# Optional - for web search
SERPAPI_API_KEY=your_serpapi_key
TAVILY_API_KEY=your_tavily_key

# Model settings
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Data settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
```

## 📊 Evaluation

Run comprehensive evaluation:

```bash
python main.py eval
```

The system evaluates responses on:
- **Faithfulness**: Accuracy to source material
- **Relevance**: Response relevance to questions
- **Tool Routing**: Correct tool selection
- **Source Quality**: Citation accuracy

## 🧪 Testing

Run the test suite:
```bash
python main.py test
```

## 🔧 API Usage

```python
import requests

# Create session
response = requests.post("http://localhost:8000/sessions", 
                        json={"user_id": "your_user_id"})
session_id = response.json()["session_id"]

# Chat
response = requests.post("http://localhost:8000/chat", json={
    "message": "What are the latest AI safety guidelines?",
    "session_id": session_id
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Tool used: {result['tool_used']}")
print(f"Sources: {result['sources']}")
```

## 🎨 Customization

### Adding New Tools

1. Create tool class in `tools.py`
2. Update router logic
3. Integrate in `chatbot.py`

### Custom Evaluation Metrics

Add evaluators in `evaluation.py` for domain-specific metrics.

## 📈 Monitoring

Track conversation statistics, tool usage patterns, and performance metrics through the built-in monitoring system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- UI powered by [Streamlit](https://streamlit.io/) and [Gradio](https://gradio.app/)
- Vector storage with [FAISS](https://github.com/facebookresearch/faiss)
- Backend with [FastAPI](https://fastapi.tiangolo.com/)

---

**Happy Research! 🤖📚**
