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

client = OpenAI()

# Define the paths for the database and pdfs.
if 'pdf_directory' not in st.session_state:
    st.session_state.pdf_directory = os.path.join("PDFs")

# We want to know how many tokens have been embedded.  It helps estimate charges.
if 'embedding_token_count' not in st.session_state:
    st.session_state.embedding_token_count = 0

st.write(f"ChromaDB version: {chromadb.__version__}")

def create_database():
    # Create a new ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create a new collection for our library
    collection = chroma_client.get_or_create_collection(name="library")
    
    return chroma_client, collection

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

def add_to_database_batch(collection, filename, pages, max_retries=3, retry_delay=1):
    # Prepare batch of texts for embedding
    texts = list(pages.values())
    page_numbers = list(pages.keys())
    
    # Get batch embeddings
    try:
        embeddings = client.embeddings.create(input=texts, model=st.session_state.embedding_model_name)
    except OpenAIError as e:
        st.warning(f'Error embedding with exception {e}.  Truncating text for batch and trying again.')
        texts = [text[:5000] for text in texts]
        try:
            embeddings = client.embeddings.create(input=texts, model=st.session_state.embedding_model_name)
        except OpenAIError as e:
            st.warning(f'Error embedding with exception {e}  Skipping this PDF.')
            return

    st.session_state.embedding_token_count += embeddings.usage.total_tokens

    # Insert each page and its embedding into the database
    ids = [f"{filename}_{page_number}" for page_number in page_numbers]
    metadatas = [{"filename": filename, "page_number": page_number} for page_number in page_numbers]
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
                st.warning(f"Database error occurred. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                st.error(f"Failed to add data to the database after {max_retries} attempts. Error: {e}")
                raise

def process_pdf(pdf_path, collection):
    filename = os.path.basename(pdf_path)
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        print(f'Failed to open file {pdf_path}.')
        return

    # Collect all the pages in the pdf and extract text
    pages = {}  # Dictionary to store text content of each page
    for page_number in range(len(pdf)):
        page = pdf.load_page(page_number)
        text_content = page.get_text() 
        if len(text_content) > 0:
            pages[page_number] = text_content

    # Check if this PDF is already in the database
    try:
        existing_pages = collection.get(
            where={"filename": filename},
            include=["metadatas"]
        )
        if len(existing_pages['metadatas']) == len(pages):
            st.write(f'Skipping {filename}, already in database.')
            return
    except InternalError as e:
        st.warning(f"Error checking existing pages for {filename}. Error: {e}")
        # Continue with adding the PDF, as we couldn't verify if it's already in the database

    st.write(f'Adding {filename}.')

    add_to_database_batch(collection, filename, pages)    

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
            n_results=5,
            include=['metadatas', 'documents', 'distances']
        )

        # # Print debug information
        # st.write("Debug information:")
        # st.write(f"Results structure: {results.keys()}")
        # for key, value in results.items():
        #     st.write(f"{key}: {type(value)}, Length: {len(value) if value is not None else 'None'}")

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
            closest_contexts.append((
                metadata['filename'],
                metadata['page_number'],
                distance,
                document
            ))

        # st.write(f"Number of results: {len(closest_contexts)}")
        # st.write(f"Distances: {results['distances']}")

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
        conversation.append({"role":"system", "content": f'Consider the following text: {c[2]}'})
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
st.sidebar.header("Configuration")
st.session_state.pdf_directory = st.sidebar.text_input("PDF Directory:", value=st.session_state.pdf_directory)

st.sidebar.header("System Information")
system_message = st.sidebar.text_input("Enter information for the system message:", "Please give a scientific answer.")

if 'chroma_client' not in st.session_state or 'collection' not in st.session_state:
    st.session_state.chroma_client, st.session_state.collection = create_database()

if st.sidebar.button(label="Populate database"):
    st.spinner("Populating database...")
    populate_database(st.session_state.collection)
    st.write('Done')

st.session_state.embedding_model_name = 'text-embedding-3-small'
st.sidebar.write(f'Embedding model: {st.session_state.embedding_model_name}.')
st.sidebar.write(f'Total tokens embedded: {st.session_state.embedding_token_count}')
st.write(f"Number of documents in collection: {st.session_state.collection.count()}")

model = st.sidebar.selectbox('Choose a model:', ('gpt-4.1-nano', 'gpt-4.1-mini'))

# Main
st.header(f"Welcome, how can I assist you today?")

# Chat Interface
user_question = st.text_area("Your Question:")
if user_question:
    # Search the pdf library to get three pages of context for GPT to use to answer the question.
    context = get_context(user_question, st.session_state.collection)

    # Go to ChatGPT now.
    answer, usage = query_gpt(system_message=system_message, context=context, question=user_question, model=model)

    # Display answer
    st.write(f"Answer: {answer}")

    # Display context for the user (they can look a the papers).
    st.write(f'The following papers are used as context on this search:')
    for c in context:
        pdf_link = f'<a href="file://{os.path.abspath(os.path.join("PDFs",c[0]))}" target="_blank">{c[0]}, page {c[1]}, distance {c[2]}</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

    st.write('Token usage information:')
    st.write(usage)
    st.write('Full Context information')
    st.write(context)