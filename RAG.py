from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_wrapper import MyCustomLLM




from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Use local embeddings - no API key needed!
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="Transformers use self-attention to model relationships."),
    Document(page_content="RAG combines retrieval with generation for grounded answers."),
    Document(page_content="FAISS is a vector database for similarity search.")
]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

docs = [
    Document(page_content="Transformers use self-attention to model relationships."),
    Document(page_content="RAG combines retrieval with generation for grounded answers."),
    Document(page_content="FAISS is a vector database for similarity search.")
]


# 1.   INTERFACE SETUP

st.set_page_config(page_title="My First AI Agent")
st.title(" My Custom AI Agent")
st.markdown("I am running a custom LLM built from scratch!")


@st.cache_resource
def load_my_model():
    return MyCustomLLM("my_llm_weights.pth", "model_config.json")


try:
    llm = load_my_model()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# 3. KNOWLEDGE BASE

st.sidebar.header(" Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload Notes (.txt)", type="txt")

if uploaded_file:

    raw_text = uploaded_file.read().decode("utf-8")

    #  Chunk
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(raw_text)
    st.sidebar.info(f"Split into {len(texts)} chunks.")



    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embeddings)

    # D. Connect it all
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2})
    )

    st.divider()


    # 4. CHAT LOOP

    query = st.text_input("Ask a question based on your notes:")

    if query:
        with st.spinner("Thinking..."):

            response = qa_chain.run(query)

        st.write("###  Answer:")
        st.write(response)

else:
    st.info(" Please upload a text file to start chatting!")
