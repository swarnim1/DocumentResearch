import os
import time
import pickle
import streamlit as st
from CustomLLM import CustomLLM
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from sentence_transformers import SentenceTransformer 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings      
from langchain_community.document_loaders import UnstructuredURLLoader

llm = CustomLLM()

st.title("Document scrambler tool")
st.sidebar.title("Enter your URL here")

urls =[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

if(process_url_clicked):
    #Loading the data
    loader  = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading Started.")
    data = loader.load()

    #Splitting the data in chunks
    text_spliter = RecursiveCharacterTextSplitter(
        separators= ['\n\n',"\n" ,',','.'],
        chunk_size = 1000 
        )
    docs = text_spliter.split_documents(data)
    main_placeholder.text("Data splitting Started.")

    #Create embedding S
    #embeddings = OpenAIEmbeddings()
    #encoder = SentenceTransformer("all-mpnet-base-v2")
    #embeddings = encoder.encode(docs)
    embeddings = HuggingFaceEmbeddings()
    #vectorstore_openai = FAISS.from_documents(docs ,  embeddings)
    vectorstore_openai = FAISS.from_documents(docs ,  embeddings)
    main_placeholder.text("Embedding vector starting.")
    time.sleep(2)

    #Save faiss index t a pickl file
    with open(file_path ,"wb") as f:
        pickle.dump(vectorstore_openai ,f)
    main_placeholder.text("Pickl file has been made.")

query = main_placeholder.text_input("Question :")
if query:
    if os.path.exists(file_path):
        with open(file_path ,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_chain_type(llm = llm , retriever= vectorstore.as_retriever(),chain_type="stuff")
            result = chain({"question" : query} , return_only_outputs= True)
            st.header("Answer")
            st.subheader(result["answer"])
    
