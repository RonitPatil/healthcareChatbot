import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
from newspaper import Article
import requests
import time


def handle_load_error(loader):
    if loader:
        st.error(f"Error occurred in loader {loader}")
    else:
        st.error("Loader is not initialized")


def get_loader_for_file(file_path):
    if file_path.endswith('.csv'):
        return CSVLoader(file_path, encoding="utf-8")
    elif file_path.endswith('.pdf'):
        return PyPDFLoader(file_path)
    elif file_path.endswith('.json'):
        return JSONLoader(file_path=file_path,
                          jq_schema='.[]',
                          text_content=False)
    elif file_path.endswith('.html'):
        return UnstructuredHTMLLoader(file_path)
    elif file_path.endswith('.md'):
        return UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError("Unsupported file type")


def populate_vector_store(uploaded_file, astra_vector_store):
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name
    try:
        loader = get_loader_for_file(tmp_file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        docs = loader.load_and_split(text_splitter)
        embedded_docs = [doc.page_content for doc in docs]
        astra_vector_store.add_texts(embedded_docs)
    except ValueError as ve:
        st.error(f"Unsupported file type: {ve}")
    except Exception as e:
        handle_load_error(loader if 'loader' in locals() else None)
    os.unlink(tmp_file_path)


def scrape_link(url, astra_vector_store):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    session = requests.Session()
    pages_content = []  # where we save the scraped article

    try:
        time.sleep(2)  # sleep two seconds for gentle scraping
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(url)
            article.download()  # download HTML of webpage
            article.parse()  # parse HTML to extract the article text
            pages_content.append({"url": url, "text": article.text})
        else:
            print(f"Failed to fetch article at {url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, length_function=len)

    all_texts, all_metadatas = [], []
    for d in pages_content:
        chunks = text_splitter.split_text(d["text"])
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": d["url"]})
    astra_vector_store.add_texts(all_texts, all_metadatas)


# if __name__ == '__main__':
#     astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])
#     populate_vector_store('Documents/budget_speech.pdf', astra_vector_store)
