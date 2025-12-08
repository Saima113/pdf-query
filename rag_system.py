import os
from typing import List, Dict, Optional
import nltk
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import hashlib
from tiktoken import get_encoding
from groq import Groq

class RAGSystem:
    def __init__(self, pdf_directory: str = "docs"):
        self.pdf_directory = pdf_directory
        
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Installing spacy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        
        self.tokenizer_tiktoken = get_encoding("cl100k_base")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        self.vector_store = None
        self._load_vector_store()
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(" WARNING: No GROQ_API_KEY found!")
            print("Get free API key from: https://console.groq.com/keys")
            print("Add to .env file: GROQ_API_KEY=your-key-here")
        self.client = Groq(api_key=api_key) if api_key else None
        print(" RAG System initialized!")

    def _load_vector_store(self):
        if os.path.exists("faiss_index"):
            try:
                print("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    "faiss_index",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(" Vector store loaded!")
            except Exception as e:
                print(f" Could not load vector store: {e}")
                self.vector_store = None

    def _preprocess_text(self, text: str) -> str:
        doc = self.nlp(text[:1000])
        processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        return processed_text

    def _get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def process_documents(self, force_reprocess: bool = False) -> None:
        documents = []
        processed_files = set()
        
        os.makedirs(self.pdf_directory, exist_ok=True)
        
        hash_file = os.path.join(self.pdf_directory, '.file_hashes')
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                processed_files = set(f.read().splitlines())
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        if not pdf_files:
            raise ValueError("No PDF files found. Please upload PDFs first.")
        
        print(f" Found {len(pdf_files)} PDF files...")
        
        for filename in pdf_files:
            file_path = os.path.join(self.pdf_directory, filename)
            file_hash = self._get_file_hash(file_path)
            
            if file_hash in processed_files and not force_reprocess:
                print(f"⏭ Skipping {filename}")
                continue
            
            try:
                print(f" Processing {filename}...")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                documents.extend(docs)
                processed_files.add(file_hash)
                print(f" Successfully processed {filename}")
            except Exception as e:
                print(f" Error: {filename}: {e}")

        if documents:
            print(f" Splitting documents...")
            chunks = self.text_splitter.split_documents(documents)
            print(f" Created {len(chunks)} chunks")
            
            if self.vector_store is None:
                self._load_vector_store()
            
            if self.vector_store is None:
                print(" Creating vector store...")
                self.vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
            else:
                print("Adding to vector store...")
                self.vector_store.add_documents(chunks)
            
            self.vector_store.save_local("faiss_index")
            
            with open(hash_file, 'w') as f:
                f.write('\n'.join(processed_files))
            
            print(" Done!")
        else:
            print(" No new documents")

    def generate_answer(self, context: str, question: str, chat_history: Optional[List] = None) -> str:
        if not self.client:
            return "ERROR: GROQ_API_KEY not set. Get free key from https://console.groq.com/keys"
        
        chat_history_text = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-3:]
            chat_history_text = "\n".join([f"Q: {h[0]}\nA: {h[1]}" for h in recent_history])
        
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
Rules:
- Only use information from the context provided
- If the answer is not in the context, say "I cannot find this information in the provided documents"
- Be concise and accurate
- Cite specific parts of the context when possible"""

        user_prompt = f"""Context from documents:
{context}

Previous conversation:
{chat_history_text}

Question: {question}

Please answer based only on the context above."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fast and free!
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}\nMake sure GROQ_API_KEY is set correctly."

    def query(self, question: str, chat_history: Optional[List] = None) -> Dict:
        if not self.vector_store:
            raise ValueError("Please process documents first using process_documents()")

        print(f" Searching for relevant documents...")
        docs = self.vector_store.similarity_search(
            question,
            k=5
        )
        
        context = "\n\n".join([f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
        print(f" Generating answer...")
        answer = self.generate_answer(context, question, chat_history)
        sources = [doc.metadata for doc in docs]

        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }

    def reset_database(self) -> None:
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        self.vector_store = None
        if os.path.exists(os.path.join(self.pdf_directory, '.file_hashes')):
            os.remove(os.path.join(self.pdf_directory, '.file_hashes'))