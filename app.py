from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from rag_system import RAGSystem

def initialize_session_state():
    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.title("PDF Document Q&A System")
    initialize_session_state()

    with st.sidebar:
        st.header("Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            os.makedirs("docs", exist_ok=True)
            
            for file in uploaded_files:
                with open(os.path.join("docs", file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            st.success(f"Uploaded {len(uploaded_files)} files")
        
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                st.session_state.rag.process_documents()
            st.success("Documents processed successfully!")
        
        if st.button("Reset Database"):
            st.session_state.rag.reset_database()
            st.session_state.chat_history = []
            st.success("Database reset successfully!")

    st.header("Chat with your documents")
    
    for human, assistant in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("assistant"):
            st.write(assistant)
    
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.chat_history.append((prompt, ""))
        
        with st.chat_message("human"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag.query(
                        prompt,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                    
                    st.write(response["answer"])
                    st.session_state.chat_history[-1] = (prompt, response["answer"])
                    
                    with st.expander("View Context and Sources"):
                        st.subheader("Context")
                        st.write(response["context"])
                        st.subheader("Sources")
                        for source in response["sources"]:
                            st.write(f"- {source.get('source', 'Unknown source')}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 