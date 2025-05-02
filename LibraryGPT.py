import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import fitz  # PyMuPDF
from openai import OpenAI, OpenAIError
import chromadb
from chromadb.config import Settings
import zlib
import shutil
from chromadb.errors import InternalError
import hashlib
from dotenv import load_dotenv
#Load the .env file which has the OpenAI API key
load_dotenv()

client = OpenAI()

# Define the paths for the database and pdfs.
if 'pdf_directory' not in st.session_state:
    st.session_state.pdf_directory = os.path.join("R:\PapersLibrary")

# We want to know how many tokens have been embedded.  It helps estimate charges.
if 'embedding_token_count' not in st.session_state:
    st.session_state.embedding_token_count = 0

def create_database():
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create a new collection for our library
    if 'chroma_collection' not in st.session_state:
        st.session_state.chroma_collection = st.session_state.chroma_client.get_or_create_collection(name="library")
    
    return
def get_embedding(text):
    if len(text) == 0:
        return None
    match st.session_state.embedding_model_name:
        case 'text-embedding-ada-002':
            text = text.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model=st.session_state.embedding_model_name)
            st.session_state.embedding_token_count += embedding.usage.total_tokens
            return embedding.data[0].embedding
        case 'text-embedding-3-small':
            text = text.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model=st.session_state.embedding_model_name)
            st.session_state.embedding_token_count += embedding.usage.total_tokens
            return embedding.data[0].embedding

def check_disk_space(path, required_space_mb=100):
    """Check if there's enough disk space available."""
    total, used, free = shutil.disk_usage(path)
    free_mb = free // (2**20)
    return free_mb > required_space_mb

def add_to_database_batch(collection, filename, pages, pdf_hash, max_retries=3, retry_delay=10):
    # Prepare batch of texts for embedding
    texts = list(pages.values())
    page_numbers = list(pages.keys())
    
    # If texts is a list with a single empty string then skip this PDF
    # This is a wierd error condition for chroma, it errors only if there is a list with one empty string.
    if not texts or not texts[0]:
        st.warning(f"Skipping {filename} because it contains no text.")
        return
    
    # Get batch embeddings
    try:
        embeddings = client.embeddings.create(input=texts, model=st.session_state.embedding_model_name)
    except OpenAIError as e:
        st.warning(f'Error embedding with exception {e}.  Truncating text for batch and trying again.')
        texts = [text[:5000] for text in texts]
        try:
            embeddings = client.embeddings.create(input=texts, model=st.session_state.embedding_model_name)
        except OpenAIError as e:
            st.warning(f'Error embedding with exception {e}.  Skipping this PDF.')
            return

    st.session_state.embedding_token_count += embeddings.usage.total_tokens

    # Insert each page and its embedding into the database
    ids = [f"{filename}_{page_number}" for page_number in page_numbers]
    metadatas = [{"filename": filename, "page_number": page_number, "pdf_hash": pdf_hash} for page_number in page_numbers]
    embeddings_list = [embedding.embedding for embedding in embeddings.data]

    for attempt in range(max_retries):
        try:
            if not check_disk_space("./chroma_db"):
                st.error("Not enough disk space available. Please free up some space and try again.")
                return

            collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=texts
            )
            break
        except InternalError as e:
            if attempt < max_retries - 1:
                st.warning(f"Database error occurred. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})\n{e}")
                time.sleep(retry_delay)
            else:
                st.error(f"Failed to add data to the database after {max_retries} attempts. Error: {e}")
                raise

def process_pdf(pdf_path, collection):
    filename = os.path.basename(pdf_path)
    
    # Calculate hash of the PDF
    with open(pdf_path, "rb") as f:
        pdf_hash = hashlib.sha256(f.read()).hexdigest()

    # Check if this PDF is already in the database
    try:
        existing_docs = collection.get(
            where={"pdf_hash": pdf_hash},
            include=["metadatas"]
        )
        if existing_docs['metadatas']:
            st.write(f'Skipping {filename}, already in database.')
            return
    except InternalError as e:
        st.warning(f"Error checking existing pages for {filename}. Error: {e}")
        # Continue with adding the PDF, as we couldn't verify if it's already in the database

    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        st.error(f'Failed to open file {pdf_path}. Error: {e}')
        return

    # Collect all the pages in the pdf and extract text
    pages = {}  # Dictionary to store text content of each page
    for page_number in range(len(pdf)):
        page = pdf.load_page(page_number)
        text_content = page.get_text() 
        if len(text_content) > 0:
            pages[page_number] = text_content

    st.write(f'Adding {filename}.')

    add_to_database_batch(collection, filename, pages, pdf_hash)    

    st.write(f'Tokens={st.session_state.embedding_token_count}')
    st.success(f"Successfully added {filename} to the database.")

def populate_database(collection):
    for filename in os.listdir(st.session_state.pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(st.session_state.pdf_directory, filename)
            process_pdf(pdf_path, collection)

def get_context(question, collection):
    # Generate the embedding for the question using OpenAI API
    question_embedding = get_embedding(question)

    try:
        # Query the collection for similar contexts
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )

        # Check if the results contain the expected keys and are not empty
        if not all(key in results and results[key] for key in ['ids', 'metadatas', 'documents', 'distances']):
            st.warning("Some expected data is missing from the query results.")
            return []

        # Assuming the results are sorted by similarity (lowest distance first)
        closest_contexts = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            distance = results['distances'][0][i]
            closest_contexts.append({
                'filename': metadata['filename'],
                'page_number': metadata['page_number'],
                'distance': distance,
                'content': document,
                'pdf_hash': metadata['pdf_hash']
            })

        return closest_contexts

    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        st.write("Debug information:")
        st.write(f"Exception type: {type(e).__name__}")
        st.write(f"Exception args: {e.args}")
        return []  # Return an empty list if there's an error

# Streamlit App
st.title("Library GPT")

def query_gpt(system_message, context, question, model='gpt-4.1-nano'):
    # Combine system message and user question
    conversation = [
        {"role": "system", "content": system_message},
    ]
    for c in context:
        conversation.append({"role":"system", "content": f'Consider the following text: {c["content"]}'})
    conversation.append({"role": "user", "content": question})

    # Make API call
    response = client.chat.completions.create(model=model,
        messages=conversation,
        temperature=0.2, # For this a lower temp is good because we want to stay faithful to the pdf context.
        max_tokens=1000  # Limit the response length
        )

    # Extract and return the assistant's reply
    assistant_reply = response.choices[0].message.content
    return assistant_reply, response.usage

# Sidebar
st.sidebar.header("Database Configuration")
st.sidebar.write(f"ChromaDB version: {chromadb.__version__}")
st.session_state.pdf_directory = st.sidebar.text_input("PDF Directory:", value=st.session_state.pdf_directory)
if 'chroma_client' not in st.session_state or 'chroma_collection' not in st.session_state:
    create_database()
st.sidebar.write(f"Number of documents in collection: {st.session_state.chroma_collection.count()}")
if st.sidebar.button(label="Populate database"):
    st.spinner("Populating database...")
    populate_database(st.session_state.chroma_collection)
    st.write('Done')
st.sidebar.write(f'Total tokens embedded: {st.session_state.embedding_token_count}')

st.sidebar.header("System Information")
system_message = st.sidebar.text_input("Enter information for the system message:", "Please give a scientific answer and cite references.")

st.sidebar.header("Query Settings")
n_results = st.sidebar.slider("Number of context results", min_value=1, max_value=50, value=5, step=1)

st.session_state.embedding_model_name = 'text-embedding-3-small'
st.sidebar.write(f'Embedding model: {st.session_state.embedding_model_name}.')

model = st.sidebar.selectbox('Choose a model:', ('gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o-mini', 'gpt-4o', 'o1-mini'))
# Query OpenAI and print the context windows size and price per megatoken for the selected model.



# Main
st.header(f"Welcome, how can I assist you today?")

# Chat Interface
user_question = st.text_area("Your Question:")
if user_question:
    # Search the pdf library to get three pages of context for GPT to use to answer the question.
    context = get_context(user_question, st.session_state.chroma_collection)

    # Go to ChatGPT now.
    answer, usage = query_gpt(system_message=system_message, context=context, question=user_question, model=model)

    # Display answer
    st.write(f"Answer: {answer}")

    # Display context for the user (they can look a the papers).
    st.write(f'The following papers are used as context on this search:')
    for c in context:
        pdf_link = f'<a href="file://{os.path.abspath(os.path.join(st.session_state.pdf_directory, c["filename"]))}" target="_blank">{c["filename"]}, page {c["page_number"]}, distance {c["distance"]:.4f}</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

    st.write('Token usage information:')
    st.write(usage)
    st.write('Full Context information')
    st.write(context)