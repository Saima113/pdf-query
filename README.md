---
title: PDF Query RAG
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
---
Advanced PDF Document Q&A System
A sophisticated Retrieval Augmented Generation (RAG) system that processes PDF documents and provides context-aware answers using advanced NLP and local models.

Features
Advanced PDF document processing with PyPDFLoader
Sophisticated text preprocessing using spaCy and NLTK
High-performance vector storage using FAISS
Local LLM-based question answering with Groq
Advanced anti-hallucination measures
Conversation history tracking
Source attribution and context transparency
User-friendly Streamlit interface
CPU acceleration support
Memory-efficient 8-bit quantization
Technical Stack
Core Components
Document Processing: PyPDF
Text Processing: spaCy, NLTK, tiktoken
Embeddings: sentence-transformers/all-mpnet-base-v2
Vector Store: FAISS
LLM: Groq
UI: Streamlit
Key Libraries
langchain & langchain-community
transformers
torch
accelerate
faiss-cpu
spacy
nltk
tiktoken
Setup
Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Download required language models:
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger
Usage
Run the Streamlit app:
streamlit run app.py
Using the interface:
Upload PDF files using the sidebar
Click "Process Documents" to create embeddings
Ask questions in the chat interface
View context and sources for each answer
Reset the database if needed
System Architecture
1. Document Processing Pipeline
PDF loading with UnstructuredPDFLoader
Text extraction and preprocessing
Chunking with RecursiveCharacterTextSplitter
Embedding generation with all-mpnet-base-v2
Vector storage in FAISS
2. Text Processing
Stop word removal
Lemmatization
Custom separators for better chunking
NLTK tokenization
spaCy linguistic features
3. Vector Storage
FAISS for efficient similarity search
CPU acceleration support
Score thresholding (0.7)
Top-k retrieval (k=5)
Persistent storage
4. Question Answering
Context-aware retrieval
Advanced prompt engineering
8-bit quantized Groq model
Conversation history integration
Source attribution
Anti-Hallucination Measures
Context Grounding:

Strict context-based answering
Multiple relevant chunks (k=5)
Score thresholding (0.7)
Source verification
Transparency:

Full context visibility
Source document tracking
Conversation history
Confidence scoring
Quality Control:

Advanced text preprocessing
Improved chunking strategy
Better prompt engineering
Temperature control
Performance Optimizations
Memory Efficiency:

8-bit model quantization
Efficient text chunking
FAISS vector storage
File hash tracking
Speed Improvements:

CPU acceleration
FAISS similarity search
Optimized text processing
Caching mechanisms
Resource Management:

Automatic device mapping
Memory-efficient embeddings
Optimized batch processing
Efficient storage
Advanced Features
Text Processing:

Lemmatization
Stop word removal
Custom separators
Advanced tokenization
Document Handling:

Change detection
Incremental updates
Error handling
File tracking
Answer Generation:

Context-aware responses
Conversation history
Source attribution
Confidence scoring
Future Improvements
Support for more document types
Document auto-upload functionality
Slack bot integration
Advanced chunking strategies
Caching improvements
Larger model support
Advanced quantization
Document summarization
Document clustering
Multi-language support
Custom embedding models
Advanced prompt templates
Troubleshooting
Memory Issues:

Enable 8-bit quantization
Reduce chunk size
Use smaller model
Clear cache
Performance Issues:

Enable CPU acceleration
Optimize chunk size
Use FAISS index
Clear unused memory
Quality Issues:

Adjust score threshold
Increase chunk overlap
Modify prompt template
Check preprocessing
About
pdf-query-nine.vercel.app
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Deployments
4
 Production 1 hour ago
+ 3 deployments
Languages
Python
100.0%
Suggested workflows
Based on your tech stack
Publish Python Package logo
Publish Python Package
Publish a Python Package to PyPI on release.
Python application logo
Python application
Create and test a Python application.
Python package logo
Python package
Create and test a Python package on multiple Python versions.
More workflows
Footer
