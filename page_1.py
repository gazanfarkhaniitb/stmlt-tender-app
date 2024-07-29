<<<<<<< HEAD
import streamlit as st
from dotenv import load_dotenv
import pickle
import fitz  # PyMuPDF
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import os
import numpy as np
import openai
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Tender App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built for analyzing tenders using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Gak and Ashar')

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read PDF using PyMuPDF and keep track of page numbers
def read_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_texts.append((page_text, page_num))
        text += page_text
    return text, page_texts

# Function to split text into chunks of max_tokens size and keep track of page numbers
def split_text(page_texts, max_tokens=3000):
    current_chunk = []
    current_length = 0
    chunks = []
    chunk_page_numbers = []
    
    for page_text, page_num in page_texts:
        sentences = page_text.split('. ')
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                chunk_page_numbers.append(page_num)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
        chunk_page_numbers.append(page_num)
    
    return chunks, chunk_page_numbers

# Function to generate embeddings using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embeddings for the given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings for each chunk."""
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    return np.array(embeddings)

# Function to normalize embeddings for cosine similarity
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = normalize_embeddings(embeddings)
    index.add(normalized_embeddings)
    return index

# Function to query embeddings and find top 3 relevant chunks
def query_embeddings(query, index, chunks, embeddings, k=3):
    query_embedding = get_embedding(query)
    normalized_query_embedding = normalize_embeddings([query_embedding])[0]
    cosine_similarities = cosine_similarity([normalized_query_embedding], embeddings)[0]
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    relevant_chunks = [(chunks[idx], idx) for idx in top_k_indices]
    return relevant_chunks

# Function to highlight search terms in text
def highlight_text(text, search_term):
    highlighted_text = text.replace(search_term, f'<mark>{search_term}</mark>')
    return highlighted_text

# Function to save chat history as PDF
def save_chat_to_pdf(chat_history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y_position = height - 40
    for i, (user_query, bot_response) in enumerate(chat_history):
        c.drawString(30, y_position, f"User: {user_query}")
        y_position -= 20
        for line in bot_response.split('\n'):
            c.drawString(30, y_position, f"Chatbot: {line}")
            y_position -= 20
            if y_position < 40:
                c.showPage()
                y_position = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit App
def main():
    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 1rem;
                padding-bottom: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            .css-1d391kg {
                padding: 1rem;
            }
            mark {
                background-color: yellow;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Chat with your PDF Tender Document 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your Tender here", type='pdf')

    if pdf is not None:
        pdf_path = pdf.name
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        text, page_texts = read_pdf(pdf_path)
        chunks, chunk_page_numbers = split_text(page_texts, max_tokens=3000)

        store_name = pdf.name[:-4]
        embeddings_file = f"{store_name}_embeddings.pkl"
        chunks_file = f"{store_name}_chunks.pkl"
        page_numbers_file = f"{store_name}_page_numbers.pkl"
        index_file = f"{store_name}_faiss.index"

        if os.path.exists(index_file) and os.path.exists(chunks_file) and os.path.exists(embeddings_file) and os.path.exists(page_numbers_file):
            index = faiss.read_index(index_file)
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            with open(embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
            with open(page_numbers_file, "rb") as f:
                chunk_page_numbers = pickle.load(f)
        else:
            embeddings = generate_embeddings_for_chunks(chunks)
            index = store_embeddings_in_faiss(embeddings)
            faiss.write_index(index, index_file)
            with open(chunks_file, "wb") as f:
                pickle.dump(chunks, f)
            with open(embeddings_file, "wb") as f:
                pickle.dump(embeddings, f)
            with open(page_numbers_file, "wb") as f:
                pickle.dump(chunk_page_numbers, f)

        search_term = st.text_input("Search in the document:")
        found_pages = []
        if search_term:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if search_term.lower() in text.lower():
                    found_pages.append(page_num + 1)
            if found_pages:
                st.write(f"Search term found on pages: {', '.join(map(str, found_pages))}")
            else:
                st.write(f"No results found for '{search_term}'")
        
        col1, col2 = st.columns([4, 2])

        with col1:
            st.subheader("PDF Viewer")
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if search_term:
                    text = highlight_text(text, search_term)
                st.write(f"### Page {page_num + 1}")
                st.markdown(text, unsafe_allow_html=True)

        with col2:
            st.subheader("Chatbot")
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            query = st.text_input("Ask questions about your PDF file:")

            if query:
                top_chunks = query_embeddings(query, index, chunks, embeddings)
                st.write("Top 3 relevant chunks:")
                for i, (chunk, idx) in enumerate(top_chunks, start=1):
                    st.write(f"**Chunk {i}:** Referenced from page {chunk_page_numbers[idx] + 1}")

                # Wrap chunks in Document objects
                docs = [Document(page_content=chunk) for chunk, idx in top_chunks]

                llm = OpenAI(model_name="gpt-3.5-turbo")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    st.write(response)

                # Update chat history
                st.session_state.chat_history.append((query, response))

            # Display chat history
            if st.session_state.chat_history:
                st.write("### Chat History")
                for i, (user_query, bot_response) in enumerate(st.session_state.chat_history):
                    st.write(f"**User:** {user_query}")
                    st.write(f"**Chatbot:** {bot_response}")

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()

            if st.button("Save Chat"):
                pdf_buffer = save_chat_to_pdf(st.session_state.chat_history)
                st.download_button("Download Chat as PDF", data=pdf_buffer, file_name="chat_history.pdf", mime="application/pdf")

if __name__ == '__main__':
    main()
=======
import streamlit as st
from dotenv import load_dotenv
import pickle
import fitz  # PyMuPDF
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import os
import numpy as np
import openai
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Tender App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built for analyzing tenders using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Gak and Ashar')

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read PDF using PyMuPDF and keep track of page numbers
def read_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_texts.append((page_text, page_num))
        text += page_text
    return text, page_texts

# Function to split text into chunks of max_tokens size and keep track of page numbers
def split_text(page_texts, max_tokens=3000):
    current_chunk = []
    current_length = 0
    chunks = []
    chunk_page_numbers = []
    
    for page_text, page_num in page_texts:
        sentences = page_text.split('. ')
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                chunk_page_numbers.append(page_num)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
        chunk_page_numbers.append(page_num)
    
    return chunks, chunk_page_numbers

# Function to generate embeddings using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embeddings for the given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings for each chunk."""
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    return np.array(embeddings)

# Function to normalize embeddings for cosine similarity
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = normalize_embeddings(embeddings)
    index.add(normalized_embeddings)
    return index

# Function to query embeddings and find top 3 relevant chunks
def query_embeddings(query, index, chunks, embeddings, k=3):
    query_embedding = get_embedding(query)
    normalized_query_embedding = normalize_embeddings([query_embedding])[0]
    cosine_similarities = cosine_similarity([normalized_query_embedding], embeddings)[0]
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    relevant_chunks = [(chunks[idx], idx) for idx in top_k_indices]
    return relevant_chunks

# Function to highlight search terms in text
def highlight_text(text, search_term):
    highlighted_text = text.replace(search_term, f'<mark>{search_term}</mark>')
    return highlighted_text

# Function to save chat history as PDF
def save_chat_to_pdf(chat_history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y_position = height - 40
    for i, (user_query, bot_response) in enumerate(chat_history):
        c.drawString(30, y_position, f"User: {user_query}")
        y_position -= 20
        for line in bot_response.split('\n'):
            c.drawString(30, y_position, f"Chatbot: {line}")
            y_position -= 20
            if y_position < 40:
                c.showPage()
                y_position = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit App
def main():
    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 1rem;
                padding-bottom: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            .css-1d391kg {
                padding: 1rem;
            }
            mark {
                background-color: yellow;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Chat with your PDF Tender Document 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your Tender here", type='pdf')

    if pdf is not None:
        pdf_path = pdf.name
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        text, page_texts = read_pdf(pdf_path)
        chunks, chunk_page_numbers = split_text(page_texts, max_tokens=3000)

        store_name = pdf.name[:-4]
        embeddings_file = f"{store_name}_embeddings.pkl"
        chunks_file = f"{store_name}_chunks.pkl"
        page_numbers_file = f"{store_name}_page_numbers.pkl"
        index_file = f"{store_name}_faiss.index"

        if os.path.exists(index_file) and os.path.exists(chunks_file) and os.path.exists(embeddings_file) and os.path.exists(page_numbers_file):
            index = faiss.read_index(index_file)
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            with open(embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
            with open(page_numbers_file, "rb") as f:
                chunk_page_numbers = pickle.load(f)
        else:
            embeddings = generate_embeddings_for_chunks(chunks)
            index = store_embeddings_in_faiss(embeddings)
            faiss.write_index(index, index_file)
            with open(chunks_file, "wb") as f:
                pickle.dump(chunks, f)
            with open(embeddings_file, "wb") as f:
                pickle.dump(embeddings, f)
            with open(page_numbers_file, "wb") as f:
                pickle.dump(chunk_page_numbers, f)

        search_term = st.text_input("Search in the document:")
        found_pages = []
        if search_term:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if search_term.lower() in text.lower():
                    found_pages.append(page_num + 1)
            if found_pages:
                st.write(f"Search term found on pages: {', '.join(map(str, found_pages))}")
            else:
                st.write(f"No results found for '{search_term}'")
        
        col1, col2 = st.columns([4, 2])

        with col1:
            st.subheader("PDF Viewer")
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if search_term:
                    text = highlight_text(text, search_term)
                st.write(f"### Page {page_num + 1}")
                st.markdown(text, unsafe_allow_html=True)

        with col2:
            st.subheader("Chatbot")
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            query = st.text_input("Ask questions about your PDF file:")

            if query:
                top_chunks = query_embeddings(query, index, chunks, embeddings)
                st.write("Top 3 relevant chunks:")
                for i, (chunk, idx) in enumerate(top_chunks, start=1):
                    st.write(f"**Chunk {i}:** Referenced from page {chunk_page_numbers[idx] + 1}")

                # Wrap chunks in Document objects
                docs = [Document(page_content=chunk) for chunk, idx in top_chunks]

                llm = OpenAI(model_name="gpt-3.5-turbo")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    st.write(response)

                # Update chat history
                st.session_state.chat_history.append((query, response))

            # Display chat history
            if st.session_state.chat_history:
                st.write("### Chat History")
                for i, (user_query, bot_response) in enumerate(st.session_state.chat_history):
                    st.write(f"**User:** {user_query}")
                    st.write(f"**Chatbot:** {bot_response}")

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()

            if st.button("Save Chat"):
                pdf_buffer = save_chat_to_pdf(st.session_state.chat_history)
                st.download_button("Download Chat as PDF", data=pdf_buffer, file_name="chat_history.pdf", mime="application/pdf")

if __name__ == '__main__':
    main()
>>>>>>> eb966d8684017c5740e03bb05f7338fe3139241c
