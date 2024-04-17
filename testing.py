import re
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai

import requests
import tempfile
import os
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

st.set_page_config(page_title="Fråga AMA Hus")

# --- ANVÄNDARUPPGIFTER ---

names = ["Denis Hujdur", "Arnela Hadzovic"]
usernames = ["Denis", "Arnela"]

# Ladda hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pk1"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "sales dashboard", "abcdef", cookie_expiry_days=30)
name, authentication_status, username = authenticator.login("Logga in", "main")

if authentication_status == False:
    st.error("Fel lösernord eller användarnamn")
if authentication_status == None:
    st.error("Ange användarnamn och lösenord")
if authentication_status:

    def extract_figure_references(text):
        # Regular expression to find figure references like "figure 3.1k"
        pattern = r"figur\s+(\d+(\.\d+)*[a-z]*)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [match[0] for match in matches]

    @st.cache(allow_output_mutation=True)
    def process_pdf(pdf_url):
        # Fetch the PDF file from the URL
        response = requests.get(pdf_url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch pre-uploaded PDF file. Status code: {response.status_code}")

        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Download the file and write it to the temporary file
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    tmp_file.write(chunk)
        
        # Extract text from the PDF content
        pdf_reader = PdfReader(tmp_file.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledgebase = faiss.FAISS.from_texts(chunks, embeddings)

        return knowledgebase

    def main():
        load_dotenv()
        st.image('./header.png', caption=None, width=500)
        st.header(":green_book: Fråga AMA Hus")
        
        # Define the URL of the PDF file on Google Drive
        pre_uploaded_pdf_url = "https://drive.google.com/uc?id=1Ch26tWqB_N-X--D83fr8MOcte6Bgplwy"
       # https://drive.google.com/file/d/1Ch26tWqB_N-X--D83fr8MOcte6Bgplwy/view?usp=sharing

        # Process the PDF file and cache the result
        with st.spinner("Laddar..."):  # Display a spinning bar while processing
            knowledgebase = process_pdf(pre_uploaded_pdf_url)

        user_question = st.text_input("Ställ en fråga")
        if user_question:
            docs = knowledgebase.similarity_search(user_question)

            llm = openai.OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Display text response
            st.write(response)

            # Extract figure references from the response
            figure_references = extract_figure_references(response)
            print("Figure References:", figure_references)

            # Display images based on figure references
            for ref in figure_references:
                image_path = f'figur_{ref}.jpg'  # Assuming images are named like "figure_3.1k.jpg"
                print("Image Path:", image_path)
                if os.path.exists(image_path):
                    st.image(image_path, caption=f'figur {ref}', use_column_width=True)
                else:
                    print("Image not found:", image_path)
                    st.write(f"Image not found: {image_path}")

    if __name__ == '__main__':
        main()
