from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.chains import RetrievalQA

st.set_page_config(page_title="Local RAG Agent")
st.title(" Local RAG Agent")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


@st.cache_resource
def load_llm():
    try:
        return Ollama(model="llama3.2", temperature=0.7)
    except Exception as e:
        st.error(f"Error: {e}")
        return None


llm = load_llm()

if llm is None:
    st.error(" Ollama not running! Start Ollama and run: `ollama pull llama3.2`")
    st.stop()
else:
    st.success("Model loaded!")

st.sidebar.header("Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload Notes (.txt)", type="txt")

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(raw_text)
    st.sidebar.success(f" {len(texts)} chunks created")

    with st.spinner("Building knowledge base..."):
        db = FAISS.from_texts(texts, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    st.divider()
    query = st.text_input(" Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.run(query)
                st.write("###  Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info(" Upload a text file to start!")