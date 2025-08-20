# Context-Aware Research Chatbot

A sophisticated conversational agent that answers domain questions using web search, local RAG (Retrieval-Augmented Generation), and mathematical tools with comprehensive source citations.

## 🚀 Features

- **Multi-Modal Intelligence**: Combines web search, local knowledge base, and mathematical calculations
- **Smart Routing**: Automatically routes queries to the most appropriate tool based on intent
- **Conversational Memory**: Maintains context across conversations with session management
- **Source Citations**: Provides detailed source attributions for all responses
- **Comprehensive Evaluation**: Built-in evaluation framework for faithfulness and groundedness
- **Multiple Interfaces**: FastAPI backend, Streamlit UI, and Gradio interface
- **Scalable Architecture**: Modular design using LangChain components

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Gradio UI     │    │   FastAPI      │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Context-Aware         │
                    │    Research Chatbot      │
                    │                          │
                    │  ┌─────────────────────┐ │
                    │  │   Query Router      │ │
                    │  └─────────┬───────────┘ │
                    │            │             │
                    │  ┌─────────▼───────────┐ │
                    │  │      Tools          │ │
                    │  │ ┌─────┐ ┌─────┐     │ │
                    │  │ │ RAG │ │ Web │     │ │
                    │  │ └─────┘ └─────┘     │ │
                    │  │ ┌─────┐             │ │
                    │  │ │Math │             │ │
                    │  │ └─────┘             │ │
                    │  └─────────────────────┘ │
                    │                          │
                    │  ┌─────────────────────┐ │
                    │  │   Memory Manager    │ │
                    │  └─────────────────────┘ │
                    └──────────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Data Layer          │
                    │                          │
                    │ ┌─────┐ ┌─────┐ ┌─────┐  │
                    │ │FAISS│ │SQLite│ │PDFs │  │
                    │ └─────┘ └─────┘ └─────┘  │
                    └──────────────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) SerpAPI or Tavily API key for web search
- PDF documents for your domain knowledge base

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd context-aware-research-chatbot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Initial Setup

```bash
python main.py setup
```

### 4. Add Your PDFs

Place your PDF documents in the `data/pdfs/` directory:

```bash
cp your-documents/*.pdf data/pdfs/
```

### 5. Process Documents

```bash
python main.py process-pdfs
```

### 6. Test System

```bash
python main.py test
```

## 🚀 Quick Start

### Option 1: Complete Workflow
```bash
python main.py all
```

### Option 2: Step by Step

1. **Start the API server:**
```bash
python main.py start-api
# Server runs on http://localhost:8000
```

2. **Start the Streamlit UI:**
```bash
python main.py start-ui
# UI available at http://localhost:8501
```

3. **Or start the Gradio UI:**
```bash
python gradio_ui.py
# UI available at http://localhost:7860
```

## 📊 Evaluation

Run comprehensive evaluation:

```bash
python main.py eval
```

Or run custom evaluation:

```bash
python evaluation.py --dataset custom_dataset.json
```

## 🔧 Configuration

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

# Vector store
VECTOR_STORE_TYPE=faiss  # or chroma
```

## 📚 Usage Examples

### API Usage

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

### Direct Python Usage

```python
from chatbot import get_chatbot

# Initialize chatbot
bot = get_chatbot()

# Create session
session_id = bot.create_session("user123")

# Chat
result = bot.chat("What is GDPR?", session_id)
print(result["response"])
```

## 🎯 Query Types & Routing

The system automatically routes queries to appropriate tools:

| Query Type | Example | Tool Used |
|------------|---------|-----------|
| Domain Knowledge | "What does GDPR say about AI?" | RAG (Local Knowledge) |
| Current Events | "Latest AI news today" | Web Search |
| Calculations | "Calculate 15% of 250000" | Math Tool |
| General AI Policy | "AI safety guidelines" | RAG → Web Search fallback |

## 🔍 Evaluation Metrics

The system evaluates responses on:

- **Faithfulness**: Does the response accurately reflect the source material?
- **Relevance**: Is the response relevant to the question?
- **Tool Routing Accuracy**: Was the correct tool used?
- **Source Quality**: Are sources properly cited and accessible?

## 📁 Project Structure

```
context-aware-research-chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                 # Main CLI interface
├── config.py              # Configuration management
├── data_processor.py      # PDF processing & vector store
├── tools.py               # Web search, math, RAG tools
├── chatbot.py             # Core chatbot logic
├── database.py            # Database models & management
├── api.py                 # FastAPI backend
├── streamlit_ui.py        # Streamlit frontend
├── gradio_ui.py           # Gradio frontend
├── evaluation.py          # Evaluation framework
├── docker-compose.yml     # Docker configuration
├── tests/                 # Test files
│   ├── test_chatbot.py
│   ├── test_tools.py
│   └── test_evaluation.py
└── data/                  # Data directory
    ├── pdfs/              # Place your PDF files here
    ├── vector_store/      # Generated vector store
    └── eval_dataset.json  # Evaluation dataset
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific tests:

```bash
python -m pytest tests/test_chatbot.py -v
```

## 🐳 Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# The services will be available at:
# - API: http://localhost:8000
# - Streamlit UI: http://localhost:8501
# - Gradio UI: http://localhost:7860
```

## 📈 Monitoring & Analytics

### Session Statistics
- Track conversations per session
- Monitor tool usage patterns
- Analyze user engagement

### Performance Metrics
- Response times by tool
- Success rates for different query types
- Source retrieval effectiveness

### Access Analytics
```bash
# Get global stats
curl http://localhost:8000/stats

# Get session stats
curl http://localhost:8000/sessions/{session_id}/stats
```

## 🔒 Security Considerations

- API keys are stored in environment variables
- Database uses SQLite by default (configure for production)
- CORS is enabled for development (configure for production)
- Session management prevents data leakage between users

## 🎛️ Customization

### Adding New Tools

1. Create tool class in `tools.py`:
```python
class CustomTool:
    def process(self, query: str) -> str:
        # Your tool logic
        return response
```

2. Update router in `tools.py`:
```python
def route(self, query: str) -> str:
    if "custom_condition" in query.lower():
        return "custom_tool"
    # ... existing logic
```

3. Integrate in `chatbot.py`:
```python
def _handle_custom_query(self, query: str, memory) -> Tuple[str, List[Dict]]:
    # Handle custom tool queries
    pass
```

### Custom Evaluation Metrics

Add custom evaluators in `evaluation.py`:

```python
def evaluate_custom_metric(self, question: str, answer: str) -> Dict[str, Any]:
    # Your custom evaluation logic
    return {"score": score, "reasoning": reasoning}
```

### UI Customization

- Modify `streamlit_ui.py` for Streamlit customizations
- Modify `gradio_ui.py` for Gradio customizations
- Both UIs consume the same FastAPI backend

## 🐛 Troubleshooting

### Common Issues

1. **Vector store not found**
   ```bash
   python main.py process-pdfs --force
   ```

2. **API connection failed**
   - Check if API server is running on port 8000
   - Verify OpenAI API key in `.env`

3. **No PDF files found**
   - Add PDF files to `data/pdfs/` directory
   - Check file permissions

4. **Memory issues with large PDFs**
   - Reduce `CHUNK_SIZE` in `.env`
   - Process PDFs in batches

5. **Evaluation failing**
   - Ensure test dataset exists
   - Check OpenAI API quota

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- UI powered by [Streamlit](https://streamlit.io/) and [Gradio](https://gradio.app/)
- Vector storage with [FAISS](https://github.com/facebookresearch/faiss) and [Chroma](https://www.trychroma.com/)
- Backend with [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the test suite for examples
- Open an issue on GitHub

---

**Happy Chatting! 🤖**