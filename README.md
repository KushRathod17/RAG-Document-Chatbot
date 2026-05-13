# RAG Document Chatbot

Upload any PDF and chat with it. Built this to learn RAG from scratch.

## What it does
- Upload a PDF from your local machine
- Ask questions about it in plain English
- Get answers with the exact page number it came from
- Full chat history so you can ask follow-up questions

## Tech used
- LangChain — connects everything together
- ChromaDB — stores document chunks as vectors
- HuggingFace (all-MiniLM-L6-v2) — converts text to embeddings
- Groq API (Llama 3.1 8B) — generates the answers
- Streamlit — the web interface

## How RAG works here
1. PDF gets split into 500 character chunks with 50 char overlap
2. Each chunk is converted to a vector and stored in ChromaDB
3. When you ask a question, it gets converted to a vector too
4. Top 3 most similar chunks are retrieved
5. Those chunks + your question go to the LLM
6. LLM answers using only those chunks — no hallucination

## Run it yourself

```bash
git clone https://github.com/KushRathod17/RAG-Document-Chatbot.git
cd RAG-Document-Chatbot
pip install -r requirements.txt
```

Add your Groq API key in a file called `rag.env`:
Then run:
```bash
streamlit run app.py
```

## What I learned building this
- How chunking strategy affects answer quality
- Why chunk overlap matters at boundaries
- Difference between keyword search (BM25) and semantic search
- How to ground LLM answers to prevent hallucination

