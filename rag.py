# Interactive RAG (Retreival Augmented Generation) Document Q&A 

import os
import time
import tempfile

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

## Langchain core classes and utilities 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langchain LLM and chaining utilities
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# text splitting and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vector store
from langchain_community.vectorstores import Chroma

# pdf file loader
from langchain_community.document_loaders import PyPDFLoader


from langchain.memory import ConversationBufferMemory
# Load the environment variable (GroqAPI-Hugging Face Token)
load_dotenv()

## Streamlit Page setup
st.set_page_config(
    page_title=" RAG Q&A with PDF uploades and chat history",
    layout="wide",
    initial_sidebar_state= "expanded"
)

st.title("RAG Q&A with PDF uploades and chat history")

st.sidebar.header("Configuration")

st.sidebar.write(

    "- Enter your GROQ API Key \n",
    "- Upload PDFs on the main page \n",
    "- Ask questions and see chat history"
)
# API Keys and Embedding setup

api_key = st.sidebar.text_input("Groq API Key", type="password")

from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
     model_kwargs={'device': 'cpu'}
    
)
if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue")
    st.stop()

# instatntiate the Groq LLM
llm= ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

# File uploader : Allow multiple PDF uploads

uploades_files = st.file_uploader(
    "Choose Pdf file(s)",
    type="pdf",
    accept_multiple_files=True,
)
# Initializing memory and history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
# A placeholder to collect all documents
all_docs = []
if uploades_files:
    # show progress spinner while loading
    with st.spinner("Loading and Splitting PDFs"):
        for pdf in uploades_files:
            # write to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path=tmp.name

            # load the pdf into a list of document objects
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    # split docs into chunks for embedding
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap =150
    )
    splits = text_splitter.split_documents(all_docs)

    # Build or load the chroma vector store (catching for performance)
    @st.cache_resource(show_spinner=False)
    def get_vectorstore(_splits):
        return Chroma.from_documents(
            _splits,
            embeddings,
            persist_directory="./chroma_index"
        )
    vectorstore = get_vectorstore(splits)
    retriever = vectorstore.as_retriever()

    # build a history aware retriever that uses past chat to refine searches.
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system","Given the chat historyand the latest user question, decide what to retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # QA chain "stuff" all retrieved docs into the llm

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. UUse the retrieved context to answer."
                   "If you don't know,say so. Keep it under three sentences. \n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # session state for chat history
    if "chathistory" not in st.session_state:
        st.session_state.chathistory = {}
    
    # getter for a session's chatMessageHistory object

    def get_history(session_id:str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id]= ChatMessageHistory()
        return st.session_state.chathistory[session_id]
    
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key ="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
  )
    # Chat UI
    session_id = st.text_input("Session ID", value= "default_session")
    user_question = st.chat_input("Your question here...")

    if user_question:
        history = get_history(session_id)
        result= conversational_rag.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = result["answer"]

        # display in streamlit new chat format
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

        with st.expander("Full chat history"):
            for msg in history.messages:
                # msg role is typically "human"
                role = getattr(msg, "role", msg.type)
                content = msg.content
                st.write(f" ** {role.title()} : ** {content}")
    else:
        # no file is uploaded yet
        st.info("Upload one or more pdfs above to begin")
