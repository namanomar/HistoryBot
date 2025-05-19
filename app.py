import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

load_dotenv()
st.set_page_config(page_title="HistoryBot", page_icon="📜")

@st.cache_resource
def load_vector_store(index_path="./faiss_index"):
    """
    Load the FAISS vector store from the specified path.
    If the index does not exist, it will create a new one.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def build_rag_chain(_vectorstore):
    """
    Build a Retrieval-Augmented Generation (RAG) chain using the provided vector store.
    This chain uses the Google Generative AI model for generating responses.
    """
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.3
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

st.title("📜 HistoryBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask a question about history...")
st.text("Ask me anything about history! I can provide information from various historical documents.")
if user_query:
    with st.spinner("Searching historical knowledge base..."):
        vectorstore = load_vector_store()
        rag_chain = build_rag_chain(vectorstore)
        result = rag_chain(user_query)

       
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("bot", result['result'], result['source_documents'][0]))

for entry in st.session_state.chat_history:
    if entry[0] == "user":
        with st.chat_message("user"):
            st.markdown(entry[1])
    elif entry[0] == "bot":
        with st.chat_message("assistant"):
            st.markdown(f"**Answer:** {entry[1]}")

            
            top_doc = entry[2]
            source_name = top_doc.metadata.get("source", "Unknown document")
            with st.expander("📚 Source"):
                st.markdown(f"**{source_name}**\n```\n{top_doc.page_content}\n```")
