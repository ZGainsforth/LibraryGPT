import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlite3
import fitz  # PyMuPDF
import openai
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Model, GPT2Tokenizer
import torch
import zlib
from dotenv import load_dotenv
load_dotenv()
# You will need a line like the following in a .env file to access the OpenAI servers.  load_dotenv() will load the value.
#OPENAI_API_KEY=<your key here>

# You will need to populate the PDFs directory with the PDFs.

# Define the paths for the database and pdfs.
pdf_directory = os.path.join("PDFs")
db_path = os.path.join("library.db")

def create_database():
                                                                                                                                                                                                                                                                                    
    # Create a new database or open connection if exists
    conn = sqlite3.connect(db_path)
                                                                                                                                                                                                                                                                                    
    # Create a cursor                                                                                                                                                                                                                                               
    cursor = conn.cursor()
                                                                                                                                                                                                                                                                                    
    # Define the table with columns: file_name, page_number, chunk_number, text_content, and embedding
    create_table_query = """CREATE TABLE IF NOT EXISTS library (
    file_name TEXT,
    page_number INTEGER,
    text_content TEXT,
    embedding BLOB);
    """
                                                                                                                                                                                                                                                                                    
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()

    # Return the path of the database to verify its creation
    db_path

def get_embedding(text):
    match st.session_state.embedding_model_name:
        case 'gpt2':
            # Initialize the GPT-2 tokenizer and model
            if 'tokenizer' not in st.session_state:
                st.session_state.tokenizer = GPT2Tokenizer.from_pretrained(st.session_state.embedding_model_name)
                st.session_state.tokenizer.pad_token = st.session_state.tokenizer.eos_token
            if 'embedding_model' not in st.session_state:
                st.session_state.embedding_model = GPT2Model.from_pretrained(st.session_state.embedding_model_name)
                # Check if a GPU is available and if not, fall back to CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Move the model to the device
                st.session_state.embedding_model.to(device)
            # Tokenize the input text
            inputs = st.session_state.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # Get the model's output
            with torch.no_grad():
                outputs = st.session_state.embedding_model(**inputs)
            # Use the last hidden state as the embedding (you can also use other layers)
            # The output shape is (batch_size, sequence_length, hidden_size)
            # We take the mean over the sequence dimension to get a single embedding vector
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding
        case 'bert-base-uncased' | 'squeezebert-uncased':
            # Initialize BERT if it hasn't been initialized
            if 'tokenizer' not in st.session_state:
                st.session_state.tokenizer = AutoTokenizer.from_pretrained(st.session_state.embedding_model_name)
            if 'embedding_model' not in st.session_state:
                st.session_state.embedding_model = AutoModel.from_pretrained(st.session_state.embedding_model_name)
                # Check if a GPU is available and if not, fall back to CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Move the model to the device
                st.session_state.embedding_model.to(device)
            text = text.replace("\n", " ")
            tokens = st.session_state.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                output = st.session_state.model(**tokens)
            # Use the mean of the last hidden state as the sentence embedding
            embedding = output.last_hidden_state.mean(dim=1)
            return embedding.cpu().numpy()
        case 'text-embedding-ada-002':
            text = text.replace("\n", " ")
            return openai.Embedding.create(input = [text], model=st.session_state.embedding_model_name)['data'][0]['embedding']

def populate_database(pdf_directory, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                pdf = fitz.open(pdf_path)
            except Exception as e:
                print(f'Failed to open file {pdf_path}.')
                continue
            
            st.write(f'Adding {filename}.')
            for page_number in range(len(pdf)):
                page = pdf.load_page(page_number)
                text_content = page.get_text()
                
                # Generate embeddings using OpenAI API
                # Note: You'll need to implement this part based on OpenAI's API documentation
                embedding = get_embedding(text_content)
                # Convert it to bytes for storage as a blob.
                embedding_blob = np.array(embedding, dtype='float32').tobytes()
                
                # Insert into SQLite database
                insert_query = """INSERT INTO library (file_name, page_number, text_content, embedding)
                                  VALUES (?, ?, ?, ?);"""
                cursor.execute(insert_query, (filename, page_number, zlib.compress(text_content.encode('utf-8')), embedding_blob))
    conn.commit()
    conn.close()

def get_context(question):
    # Generate the embedding for the question using OpenAI API (placeholder)
    question_embedding = get_embedding(question)
    question_embedding = np.array(question_embedding).reshape(1, -1)  # Reshape for cosine_similarity
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch all embeddings from the database
    cursor.execute("SELECT file_name, page_number, text_content, embedding FROM library")
    rows = cursor.fetchall()
    
    # Calculate cosine similarity
    db_embeddings = [np.frombuffer(row[3], dtype='float32') for row in rows]
    db_embeddings = np.vstack(db_embeddings)
    similarities = cosine_similarity(question_embedding, db_embeddings)
    
    # Get the indices of the three most similar embeddings
    top_indices = np.argsort(similarities[0])[-3:][::-1]
    
    # Fetch the corresponding file name, page number and text.
    closest_contexts = [(rows[i][0], rows[i][1], zlib.decompress(rows[i][2])) for i in top_indices]
    st.write(closest_contexts)
    
    conn.close()
    
    return closest_contexts

# Streamlit App
st.title("Library GPT")

def query_gpt(system_message, context, question, model='gpt-3.5-turbo-16k'):
    # Combine system message and user question
    conversation = [
        {"role": "system", "content": system_message},
        # {"role": "system", "content": context},
        # {"role": "user", "content": question}
    ]
    for c in context:
        conversation.append({"role":"system", "content": f'Consider the following text: {c[2]}'})
    conversation.append({"role": "user", "content": question})

    # Make API call
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation,
        temperature=0.2, # For this a lower temp is good because we want to stay faithful to the pdf context.
        max_tokens=2000  # Limit the response length
    )
    
    # Extract and return the assistant's reply
    assistant_reply = response['choices'][0]['message']['content']
    return assistant_reply

# Sidebar
st.sidebar.header("System Information")
system_message = st.sidebar.text_input("Enter information for the system message:", "Please give a scientific answer.")

if st.sidebar.button(label="Populate database"):
    if os.path.exists(db_path):
        os.remove(db_path)
    create_database()
    st.write(f'New database created in {db_path}.')
    st.spinner("Populating database...")
    populate_database(pdf_directory, db_path)
    st.write('Done')

st.session_state.embedding_model_name = 'gpt2'
st.sidebar.write(f'Embedding model: {st.session_state.embedding_model_name}.')
# embedding_algorithm = st.sidebar.selectbox('Choose an embedding algorithm:', ('squeezebert-uncased', 'bert-base-uncased', 'text-embedding-ada-002'))
# if 'bert' in embedding_algorithm.lower():

model = st.sidebar.selectbox('Choose a model:', ('gpt-3.5-turbo-16k', 'gpt-4'))

# Main
st.header(f"Welcome, how can I assist you today?")

# Chat Interface
user_question = st.text_area("Your Question:")
if user_question:
    # Search the pdf library to get three pages of context for GPT to use to answer the question.
    context = get_context(user_question)

    # Go to ChatGPT now.
    answer = query_gpt(system_message=system_message, context=context, question=user_question, model=model)
    
    # Display answer
    st.write(f"Answer: {answer}")

    # Display context for the user (they can look a the papers).
    st.write(f'The following papers are used as context on this search:')
    for c in context:
        # st.write(f'{c[0]}, page {c[1]}')
        pdf_link = f'<a href="file://{os.path.abspath(os.path.join("PDFs",c[0]))}" target="_blank">{c[0]}, page {c[1]}</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)
