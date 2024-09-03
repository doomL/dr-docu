from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st
import os
from getModels import get_models 

load_dotenv()

models = get_models()

# Function to get the API key
def get_api_key():
    # Try to get the API key from st.secrets first
    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        return groq_api_key
    except Exception as e:
        print(e)


# Function for API configuration at sidebar
def sidebar_api_key_configuration():
    groq_api_key = get_api_key()
    if groq_api_key == '':
        st.sidebar.warning('Inserisci la API Key 🗝️')
        st.session_state.prompt_activation = False
    elif (groq_api_key.startswith('gsk_') and (len(groq_api_key) == 56)):
        st.sidebar.success('Pronto', icon='️👉')
        st.session_state.prompt_activation = True
    else:
        st.sidebar.warning('Inserisci la API Key corretta 🗝️!', icon='⚠️')
        st.session_state.prompt_activation = False
    return groq_api_key



def sidebar_groq_model_selection():
    
    
    if not models:
        st.sidebar.error("Nessun Modello Disponibile.")
        return None

    # Extract model IDs for the selectbox options
    model_options = [model['id'] for model in models]

    st.sidebar.subheader("Scegli il Modello")
    selected_model_id = st.sidebar.selectbox('Select the Model', model_options, label_visibility="collapsed")
    


    return selected_model_id


# Read PDF data
def read_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    print(f"Number of text chunks created: {len(text_chunks)}")

    return text_chunks


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"Using embedding model: {embeddings.model}")
    return embeddings


# Create vectorstore
def create_vectorstore(pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks

    # Check if text_chunks is empty
    if not text_chunks:
        raise ValueError("No text chunks were created. Please check the input PDF documents.")

    embeddings = get_embedding_function()  # Get the embedding function

    # Check if embeddings are being generated
    sample_embedding = embeddings.embed_query(text_chunks[0])
    if not sample_embedding:
        raise ValueError("Failed to generate embeddings. Please check the embedding function.")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def update_message_history(user_message, ai_response):
    st.session_state.message_history.append({"role": "user", "content": user_message})
    st.session_state.message_history.append({"role": "ai", "content": ai_response})

# Function to generate prompt with limited history
def generate_prompt_from_history(history, length):
    prompt = ""
    # Use only the last 'length' messages
    history_to_use = history[-length:]
    for message in history_to_use:
        prompt += f"{message['role']}: {message['content']}\n"
    return prompt

# Get response from llm of user asked question
# def get_llm_response(llm, prompt, question):
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
#     response = retrieval_chain.invoke({'input': question})
#     return response
def get_llm_response(llm, prompt, question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response