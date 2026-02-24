import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain modules
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables (Get API key and DB URL from .env file)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "pdf_rag_collection" # Collection name to distinguish data within the pgvector table

# Validate environment variables
if not OPENAI_API_KEY or not DATABASE_URL:
    st.error("Error: Please set OPENAI_API_KEY and DATABASE_URL in your .env file.")
    st.stop()

# Configure Streamlit UI
st.set_page_config(page_title="PDF RAG Bot", page_icon="📄", layout="wide")
st.title("📄 PDF-based RAG Chatbot (PostgreSQL + pgvector)")

# 2. Vector Store initialization function
@st.cache_resource
def get_vectorstore():
    """Creates a VectorStore object connected to PostgreSQL (pgvector)."""
    # Use OpenAI default embedding model (Dimension: 1536)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True, # Store metadata in JSONB format to improve search efficiency
    )
    return vectorstore

# 3. PDF processing and DB storage function
def process_and_save_pdf(uploaded_file):
    """Splits the uploaded PDF into text chunks and saves them to the Vector DB."""
    # PyPDFLoader requires a file path, so save it as a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load PDF document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Text splitting (Chunking)
        # chunk_size: size of the chunks, chunk_overlap: number of overlapping characters to maintain context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # Add documents to Vector DB 
        # (This process calls the OpenAI API to create embeddings and INSERTs them into Postgres)
        vectorstore = get_vectorstore()
        vectorstore.add_documents(docs)
        
        return True
    except Exception as e:
        st.error(f"Error occurred while processing PDF: {e}")
        return False
    finally:
        os.remove(tmp_path) # Delete temporary file

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Please upload a PDF file to analyze.", type=["pdf"])
    
    if uploaded_file and st.button("Save to DB"):
        with st.spinner("Analyzing document, converting to vectors, and saving to DB..."):
            if process_and_save_pdf(uploaded_file):
                st.success("✅ Saved to DB! You can now ask questions.")

# --- Main Screen: Chat Interface ---
st.header("2. Ask Questions about the Document")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question input
if user_query := st.chat_input("Ask anything about the content of the PDF document."):
    # Display user question on screen and save to session
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Execute RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching for answers in the document..."):
            vectorstore = get_vectorstore()
            
            # Setup Retriever to fetch the top 4 most similar documents to the question from the DB
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # Setup LLM model (ChatGPT)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            # Define prompt template
            prompt_template = ChatPromptTemplate.from_template(
                """You are an AI assistant that answers questions based on the provided documents.
                Use the following Context to clearly answer the user's Question.
                If the answer is not found in the Context, say "I cannot find the relevant information in the provided documents." and do not make up information.
                
                Context: {context}
                
                Question: {question}
                
                Answer:"""
            )
            
            # Function to combine retrieved documents into a single string
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Configure LangChain LCEL (LangChain Expression Language) chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            # Execute chain and output result
            response = rag_chain.invoke(user_query)
            st.markdown(response)
            
            # Save chatbot response to session
            st.session_state.messages.append({"role": "assistant", "content": response})
