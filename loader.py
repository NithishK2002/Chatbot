from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from os import getenv
import os
OPENAI_API_KEY = getenv('OPENAI_API_KEY')

def scrape_and_store(urls, output_directory, flag):
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    if flag == 1:
        for idx, url in enumerate(urls):
            if not url.startswith("https://") and not url.startswith("http://"):
                raise ValueError(f"Invalid URL: {url}. URL must start with 'https://' or 'http://'")

            loader = WebBaseLoader(url)
            document = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            document_chunks = text_splitter.split_documents(document)
            # vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(), persist_directory=output_directory)  # Specify persist_directory parameter here
            vector_store = FAISS.from_documents(document_chunks,OpenAIEmbeddings())
            vector_store1 = vector_store.save_local(output_directory)
            # vector_store.persist()
    else:
        # files = os.listdir(urls)

    # Loop through each file in the directory
        # for file in files:
        # Check if the file is a PDF
            # if file.endswith('.pdf'):
            # Get the full path to the PDF file
                # pdf_file_path = os.path.join(urls, file)
                # print(f"Reading PDF file: {pdf_file_path}")
        loader = PyPDFDirectoryLoader(urls)
        document=loader.load()
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks= text_splitter.split_documents(document)
        # vector_score=Chroma.from_documents(document_chunks, OpenAIEmbeddings(), persist_directory=output_directory)
        vector_score = FAISS.from_documents(document_chunks,OpenAIEmbeddings())
        vector_store1 = vector_score.save_local(output_directory)
        # vector_score.persist()

if __name__ == "__main__":
    urls = ["https://en.wikipedia.org/wiki/List_of_schemes_of_the_government_of_India"]  # Add your list of URLs here
    output_directory = "faiss_url_1"
    scrape_and_store(urls, output_directory,1)
    # output_directory="faiss_pdf_1"
    # input_directory=os.getcwd()+"/pdf"
    # scrape_and_store(input_directory,output_directory,2)
