import streamlit as st
import os
from langchain_community.document_loaders import JSONLoader

from utils.initialize_vector_store import initialize_vector_store


def handle_load_error(loader):
    if loader:
        st.error(f"Error occurred in loader {loader}")
    else:
        st.error("Loader is not initialized")


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["source"] = record.get("link")
    return metadata


def populate_vector_store(file_path, astra_vector_store):
    try:
        loader = JSONLoader(file_path=file_path,
                            jq_schema='.[]',
                            text_content=False,
                            metadata_func=metadata_func)
        docs = loader.load()
        embedded_docs = [doc.page_content for doc in docs]
        astra_vector_store.add_texts(embedded_docs)
        print(f'Loaded file {file_path}')
    except ValueError as ve:
        st.error(f"Unsupported file type: {ve}")
    except Exception as e:
        handle_load_error(loader if 'loader' in locals() else None)


if __name__ == '__main__':
    astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])
    folder_path = '../SWM'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        populate_vector_store(file_path, astra_vector_store)
    folder_path = '../data'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        populate_vector_store(file_path, astra_vector_store)
