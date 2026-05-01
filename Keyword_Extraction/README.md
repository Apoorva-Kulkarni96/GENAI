# 🔑 Keyword Extraction Chatbot

Extract keywords from natural language using RAKE + a local LLM via Ollama.  
Two implementations — same logic, different tool-calling approach.

## 🗂 Versions

| | V1 — Raw Schema | V2 — LangChain |
|---|---|---|
| LLM Client | `openai` (Ollama endpoint) | `ChatOllama` |
| Tool Definition | Handwritten JSON schema | `@tool` decorator |
| Tool Dispatch | Manual `json.loads` + if/else | `tools_map[name].invoke(args)` |
| Ollama Compatible | ❌ Schema mismatch error | ✅ Works |


## 🛠 Tech Stack

### V1
- [Ollama](https://ollama.com) — local LLM server
- [openai](https://pypi.org/project/openai/) — OpenAI-compatible client
- [rake-nltk](https://pypi.org/project/rake-nltk/) — keyword extraction
- [gradio](https://www.gradio.app) — chat UI

### V2
- [Ollama](https://ollama.com) — local LLM server
- [langchain](https://python.langchain.com) — LLM framework
- [langchain-ollama](https://pypi.org/project/langchain-ollama/) — Ollama integration
- [rake-nltk](https://pypi.org/project/rake-nltk/) — keyword extraction
- [gradio](https://www.gradio.app) — chat UI

---

## ⚙️ Prerequisites

- Ollama installed and running on `http://localhost:11434`
- Model pulled:
```bash
  ollama pull devstral:24b
```
- Python 3.10+

---

## 🚀 Setup

```bash
# Clone
git clone https://github.com/yourname/keyword-extraction.git
cd keyword-extraction

# Install dependencies

# V1
pip install gradio openai rake-nltk

# V2
pip install gradio langchain langchain-ollama rake-nltk

# NLTK data (both versions)
python -m nltk.downloader stopwords punkt
```
