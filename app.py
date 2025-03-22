import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq Demo")

# Model selection
selected_model = st.selectbox("Select a Groq Model", ["mixtral-8x7b-32768", "llama3-8b-8192", "llama3-70b-8192"])

# Load embeddings
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load documents from website
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/observability/")
    raw_docs = st.session_state.loader.load()
    
    st.write(f"Loaded {len(raw_docs)} documents from the website.")  # Debugging info

    # Extract text and clean it
    clean_docs = []
    for doc in raw_docs:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        text = soup.get_text(strip=True)  # Remove HTML tags
        if text:
            clean_docs.append(text)

    st.write(f"Cleaned {len(clean_docs)} valid documents.")  # Debugging info

    # Ensure there are valid documents
    if clean_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        final_docs = text_splitter.split_text("\n".join(clean_docs))  

        if final_docs:
            st.session_state.vectors = FAISS.from_texts(final_docs, st.session_state.embeddings)
            st.write("FAISS vector store successfully created ✅")
        else:
            st.error("Not enough text for FAISS vector store.")
    else:
        st.error("No valid text extracted from the webpage.")

# Initialize LLM with selected model
llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
<context>
{context}
<context>
Questions: {input}
"""
)

# Ensuring FAISS is created before proceeding
if "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User input prompt
    user_prompt = st.text_input("Input your prompt here")

    if user_prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")
        st.write(response['answer'])

        # Show document sources
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
