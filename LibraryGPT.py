import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlite3
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
load_dotenv()
# You will need a line like the following in a .env file to access the OpenAI servers.  load_dotenv() will load the value.
#OPENAI_API_KEY=<your key here>

# You will need to populate the PDFs directory with the PDFs.

# Define the paths for the database and pdfs.
pdf_directory = os.path.join("PDFs")
db_path = os.path.join("library.db")

def create_database():
    # Define the path for database
    db_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'LibraryGPT', 'library.db')
                                                                                                                                                                                                                                                                                    
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

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def populate_database(pdf_directory, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            pdf = fitz.open(pdf_path)
            
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
                cursor.execute(insert_query, (filename, page_number, text_content, embedding_blob))
            st.write(f'Added {filename}.')
                
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
    closest_contexts = [(rows[i][0], rows[i][1], rows[i][2]) for i in top_indices]
    
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

if st.sidebar.button(label="Reset database"):
    os.remove(db_path)
    create_database()
    st.write(f'New database created in {db_path}.')

if st.sidebar.button(label="Populate database"):
    st.spinner("Populating database...")
    populate_database(pdf_directory, db_path)
    st.write('Done')

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
        st.write(f'{c[0]}, page {c[1]}')
