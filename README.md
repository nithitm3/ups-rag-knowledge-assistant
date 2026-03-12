# UPS Knowledge Assistant – Retrieval Augmented Generation (RAG) System

## Overview

This project implements a **Retrieval Augmented Generation (RAG) based chatbot** that allows users to ask questions about the **UPS Sustainability Report**.
The system retrieves relevant information from the document and generates **grounded answers** using a Large Language Model (LLM).

The goal of this system is to ensure that answers are **based strictly on the provided document**, preventing hallucinations and improving factual reliability.

The application includes:

* Document ingestion and preprocessing
* Markdown conversion for structured parsing
* Semantic chunking
* Embedding generation
* Hybrid retrieval (Vector Search + BM25)
* Two-stage reranking (Cosine Similarity + Cross-Encoder)
* LLM-based answer generation
* Chatbot UI using Chainlit

If the system cannot find an answer in the document, it returns the fallback response:

```
I don't know based on the provided information.
```

---

# System Architecture

```
User Query
      ↓
Hybrid Retrieval
(Vector Search + BM25)
      ↓
Cosine Similarity Reranking
      ↓
Top Candidate Chunks
      ↓
Cross-Encoder Reranking
      ↓
Top Relevant Chunks
      ↓
LLM (Groq)
      ↓
Grounded Answer
```

This architecture ensures the LLM receives **highly relevant context**, improving accuracy and reducing hallucinations.

---

# Project Structure

```
project
│
├── app.py                 # Chainlit chatbot interface
├── rag_pipeline.py        # RAG pipeline (retrieval + reranking + LLM)
├── document_ingestion.py  # Document loading, markdown conversion, chunking
├── vector_store.py        # Vector database management
├── uploads/
│   └── documents.pdf      # Input document
├── chroma_db/             # Persistent vector database
├── requirements.txt
└── README.md
```

---

# How to Run the Solution

## 1. Clone the repository

```
git clone <repository-url>
cd <project-folder>
```

---

## 2. Install dependencies

```
pip install -r requirements.txt
```

Main dependencies:

* langchain
* chromadb
* sentence-transformers
* chainlit
* langchain-groq
* scikit-learn
* docling

---

## 3. Configure API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

This key is required to access the Groq LLM service.

---

## 4. Run the chatbot

```
chainlit run app.py
```

Open the application in your browser:

```
http://localhost:8000
```

---

# Embedding Model Choice

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

Reasons for selecting this model:

* Lightweight and efficient
* Good semantic similarity performance
* Fast embedding generation
* Widely used in RAG systems

Embeddings convert document chunks into vector representations that allow **semantic search** over the document.

---

# Vector Database Choice

**Database:** ChromaDB

Reasons for selecting ChromaDB:

* Lightweight and easy to integrate
* Native support with LangChain
* Persistent local storage
* Fast vector similarity search

ChromaDB stores the document embeddings and enables efficient vector-based retrieval.

---

# Document Processing and Chunking Strategy

The document is first converted from **PDF to Markdown** to preserve structural information such as headings and sections.

Two chunking strategies are used.

## 1. Markdown Header Chunking

`MarkdownHeaderTextSplitter` splits the document based on semantic structure such as:

* headings
* subheadings
* sections

This ensures chunks align with logical document boundaries.

## 2. Semantic Chunking

`SemanticChunker` further splits large sections into smaller chunks based on semantic similarity.

Benefits:

* maintains contextual coherence
* prevents overly large chunks
* improves retrieval accuracy

Each chunk is stored with metadata such as:

```
{
  "source": "documents.md",
  "section": "Sustainability Strategy",
  "chunk_id": 34
}
```

This metadata allows the chatbot UI to display **source references**.

---

# Retrieval Strategy

The system uses **Hybrid Retrieval**, combining two complementary approaches.

## Vector Search

Uses embeddings to retrieve semantically similar document chunks.

Advantages:

* captures contextual meaning
* retrieves conceptually related content

## BM25 Keyword Search

Uses a keyword-based ranking algorithm.

Advantages:

* captures exact keyword matches
* effective for factual queries

Combining both methods improves **recall and robustness**.

---

# Two-Stage Reranking Strategy

After retrieval, results are reranked using a **two-stage pipeline**.

## Stage 1 — Cosine Similarity Reranking

Cosine similarity is used to measure semantic similarity between query embeddings and document embeddings.

Steps:

1. Embed the query
2. Embed retrieved chunks
3. Compute cosine similarity scores
4. Rank chunks
5. Select top candidates

This step is **fast and efficient** and reduces the candidate pool.

---

## Stage 2 — Cross-Encoder Reranking

A **cross-encoder model** is used to further refine the ranking.

Unlike embedding similarity, the cross-encoder evaluates the **query and document together**.

Instead of comparing:

```
Query Embedding ↔ Document Embedding
```

the model evaluates:

```
[Query + Document] → Relevance Score
```

This allows the model to capture deeper relationships between the query and document text.

### Cross-Encoder Model Used

```
BAAI/bge-reranker-large
```

This model is specifically designed for **document ranking tasks** and significantly improves retrieval accuracy.

---

## Final Retrieval Pipeline

```
Initial Retrieval (Vector + BM25)
↓
Cosine Similarity Reranking
↓
Top 8 Candidate Chunks
↓
Cross-Encoder Reranking
↓
Top 3 Relevant Chunks
↓
LLM Answer Generation
```

This approach balances **speed and accuracy**, ensuring only the most relevant context is provided to the LLM.

---

# Large Language Model (LLM)

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct`
**Provider:** Groq

Reasons for choosing this model:

* Fast inference on Groq infrastructure
* Strong reasoning ability
* Good performance for question answering
* Large context window

The LLM generates answers strictly based on the retrieved context.

---

# Prompt Design

The prompt enforces **grounded generation** with strict rules:

* Do not use external knowledge
* Do not hallucinate
* Only answer from retrieved context
* Return fallback response if information is missing

Fallback response:

```
"I don't know based on the provided information."
```

---

# User Interface

The chatbot interface is implemented using **Chainlit**.

Features include:

* ChatGPT-style conversational interface
* Streaming responses
* Source citations
* Metadata display for retrieved chunks
* Clean chat-based interaction

Example interaction:

```
User:
What is UPS sustainability strategy?

Assistant:
UPS sustainability strategy focuses on reducing carbon emissions
and investing in alternative fuel vehicles.

Sources:
documents.md | Sustainability Strategy | chunk 34
documents.md | Environmental Impact | chunk 52
```

---

# Limitations

### 1. Single Document Indexing

Currently the system indexes only a single document.

### 2. Cross-Encoder Latency

Cross-encoder models are computationally heavier than embedding models.

### 3. Context Window Limits

Very large documents may require advanced context compression.

### 4. Metadata Dependency

Section metadata depends on document structure and headings.

---

# Possible Improvements

### Multi-Document Knowledge Base

Allow indexing multiple documents and building a larger knowledge base.

### Query Rewriting

Use LLM-based query rewriting to improve retrieval quality.

### Context Compression

Use techniques such as:

* Max Marginal Relevance (MMR)
* Context filtering

### Retrieval Evaluation

Add evaluation metrics such as:

* retrieval precision
* answer relevance
* latency

### Highlight Retrieved Context

Highlight the exact text chunks used to generate answers.

### Lightweight Rerankers

Use smaller rerankers for faster inference in production environments.

---

# Example Interaction

**User**

```
What is UPS sustainability strategy?
```

**Assistant**

```
UPS sustainability strategy focuses on reducing carbon emissions
and investing in alternative fuel delivery vehicles.
```

**Sources**

```
documents.md | Sustainability Strategy | chunk 34
documents.md | Environmental Impact | chunk 52
```

---

# Conclusion

This project demonstrates a **complete RAG-based question answering system** combining:

* semantic search
* keyword retrieval
* hybrid retrieval
* multi-stage reranking
* LLM grounded generation
* conversational UI

By combining **hybrid retrieval, cosine similarity filtering, and cross-encoder reranking**, the system ensures that answers remain **accurate, grounded, and context-aware**.

The architecture is modular and can be extended to support larger knowledge bases and more advanced retrieval strategies.
