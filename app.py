import os 
import streamlit as st 
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # used to do vector store
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #used to do vector embeddings 
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("groq_api_key")
os.environ["GOOGLE_API_KEY"]=os.getenv("google_api_key")

st.title("Gemma model Document QNA")
llm = ChatGroq(groq_api_key= groq_api_key , model_name = "gemma-7b-it")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embeddings():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("What do you want to ask from the documents ? ")

vector_embeddings()
st.write("Vector Store Created")

import time 
if prompt1:
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vector_store.as_retriever()  # Corrected attribute name
    retrieval_chain = create_retrieval_chain(retriever , document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt1})
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
