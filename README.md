# Advanced PDF Document Q&A System

A sophisticated Retrieval Augmented Generation (RAG) system that processes PDF documents and provides context-aware answers using advanced NLP and local models.

## Features

* Advanced PDF document processing with PyPDFLoader
* Sophisticated text preprocessing using spaCy and NLTK
* High-performance vector storage using FAISS
* Local LLM-based question answering with Groq
* Advanced anti-hallucination measures
* Conversation history tracking
* Source attribution and context transparency
* User-friendly Streamlit interface
* CPU acceleration support
* Memory-efficient 8-bit quantization

## Technical Stack

### Core Components

* Document Processing: PyPDF
* Text Processing: spaCy, NLTK, tiktoken
* Embeddings: sentence-transformers/all-mpnet-base-v2
* Vector Store: FAISS
* LLM: Groq
* UI: Streamlit

### Key Libraries

* langchain & langchain-community
* transformers
* torch
* accelerate
* faiss-cpu
* spacy
* nltk
* tiktoken

## Setup

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required language models:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Using the interface:
   * Upload PDF files using the sidebar
   * Click "Process Documents" to create embeddings
   * Ask questions in the chat interface
   * View context and sources for each answer
   * Reset the database if needed

## System Architecture

### 1. Document Processing Pipeline

* PDF loading with UnstructuredPDFLoader
* Text extraction and preprocessing
* Chunking with RecursiveCharacterTextSplitter
* Embedding generation with all-mpnet-base-v2
* Vector storage in FAISS

### 2. Text Processing

* Stop word removal
* Lemmatization
* Custom separators for better chunking
* NLTK tokenization
* spaCy linguistic features

### 3. Vector Storage

* FAISS for efficient similarity search
* CPU acceleration support
* Score thresholding (0.7)
* Top-k retrieval (k=5)
* Persistent storage

### 4. Question Answering

* Context-aware retrieval
* Advanced prompt engineering
* 8-bit quantized Groq model
* Conversation history integration
* Source attribution

## Anti-Hallucination Measures

1. Context Grounding:
   * Strict context-based answering
   * Multiple relevant chunks (k=5)
   * Score thresholding (0.7)
   * Source verification

2. Transparency:
   * Full context visibility
   * Source document tracking
   * Conversation history
   * Confidence scoring

3. Quality Control:
   * Advanced text preprocessing
   * Improved chunking strategy
   * Better prompt engineering
   * Temperature control

## Performance Optimizations

1. Memory Efficiency:
   * 8-bit model quantization
   * Efficient text chunking
   * FAISS vector storage
   * File hash tracking

2. Speed Improvements:
   * CPU acceleration
   * FAISS similarity search
   * Optimized text processing
   * Caching mechanisms

3. Resource Management:
   * Automatic device mapping
   * Memory-efficient embeddings
   * Optimized batch processing
   * Efficient storage

## Advanced Features

1. Text Processing:
   * Lemmatization
   * Stop word removal
   * Custom separators
   * Advanced tokenization

2. Document Handling:
   * Change detection
   * Incremental updates
   * Error handling
   * File tracking

3. Answer Generation:
   * Context-aware responses
   * Conversation history
   * Source attribution
   * Confidence scoring

## Future Improvements

* Support for more document types
* Document auto-upload functionality
* Slack bot integration
* Advanced chunking strategies
* Caching improvements
* Larger model support
* Advanced quantization
* Document summarization
* Document clustering
* Multi-language support
* Custom embedding models
* Advanced prompt templates

## Troubleshooting

1. Memory Issues:
   * Enable 8-bit quantization
   * Reduce chunk size
   * Use smaller model
   * Clear cache

2. Performance Issues:
   * Enable CPU acceleration
   * Optimize chunk size
   * Use FAISS index
   * Clear unused memory

3. Quality Issues:
   * Adjust score threshold
   * Increase chunk overlap
   * Modify prompt template
   * Check preprocessing
