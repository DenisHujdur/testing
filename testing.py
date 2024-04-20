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
from google.oauth2 import service_account
from googleapiclient.discovery import build

import io
from googleapiclient.http import MediaIoBaseDownload

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
        # Regular expression to find figure references like "figure 3.1k" or "figure 3/14" or "figure 3-14" or "figure 3.14"
        pattern = r"figur\s+([\w/.-]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return matches




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

    def fetch_image_from_drive(file_name):
        # Initialize the Google Drive API (remaining code unchanged)
        SCOPES = ['https://www.googleapis.com/auth/drive']
        SERVICE_ACCOUNT_FILE = 'pdf-python-420718-4a9ce0b58697.json'

        creds = None
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        service = build('drive', 'v3', credentials=creds)

        # Search for the image files in Google Drive with the specified name
        results = service.files().list(q=f"mimeType='image/jpeg' and name contains '{file_name}'").execute()
        items = results.get('files', [])

        if not items:
            raise ValueError(f"No image file found with the name containing: {file_name}")

        # Filter the files based on the specified extension
        matching_files = [file for file in items if file['name'].endswith('.jpg')]

        if not matching_files:
            raise ValueError(f"No image file found with the name containing '{file_name}' and ending with '.jpg'")

        # Get the file ID of the first matching file
        file_id = matching_files[0]['id']


# pdf-python-420718-4a9ce0b58697.json
    


        # Fetch image content using file ID
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # Return image content
        return fh.getvalue()


    def main():
        load_dotenv()
        st.image('./header.png', caption=None, width=500)
        st.header(":green_book: Fråga AMA Hus")
        
        # Define the URL of the PDF file on Google Drive
        pre_uploaded_pdf_url = "https://drive.google.com/uc?id=1OtVY2AlnSk-hyjoVfRNl1u9qxyRKI2Ho"
       #https://drive.google.com/file/d/1OtVY2AlnSk-hyjoVfRNl1u9qxyRKI2Ho/view?usp=drive_link

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
                try:
                    # Use the figure reference from the text as caption
                    # st.write(f"figur {ref}")
                    # Fetch image from Google Drive based on figure reference
                    image_content = fetch_image_from_drive(f"figur_{ref}.jpg")
                    st.image(image_content, caption=f'figur {ref}', width=500)
                except ValueError as e:
                    print(str(e))
                    st.write(f"Failed to fetch image for reference {ref}")

    if __name__ == '__main__':
        main()
