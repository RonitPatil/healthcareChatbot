import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

from utils.data_loader import populate_vector_store, scrape_link
from utils.initialize_vector_store import initialize_vector_store
from utils.create_chains import create_agent_executor


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def main():
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])

    st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
    st.header('Agent Healthcare Chatbot')
    with st.sidebar:
        st.header("Upload Section")
        with st.container(border=True):
            st.markdown("### Upload Files")
            uploaded_files = st.file_uploader("Upload a file",
                                              accept_multiple_files=True,
                                              type=['csv', 'pdf', 'json', 'html', 'md'],
                                              label_visibility='hidden')
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    populate_vector_store(uploaded_file, astra_vector_store)
                    st.success(f"Processed file {uploaded_file.name}. You may ask me questions about the file now.")

        with st.container(border=True):
            st.markdown("### Submit Links")
            new_link = st.text_input("Enter a link",
                                     key="new_link",
                                     placeholder="Paste your link here...",
                                     label_visibility='hidden')
            if st.button("Submit Link", key="submit_link") or new_link:
                try:
                    if new_link != "":
                        scrape_link(new_link, astra_vector_store)
                        st.success("Link successfully scraped and processed!")
                    else:
                        st.error("Please enter a link")
                except Exception as e:
                    st.error(f"Failed to scrape link: {e}")

    for message in st.session_state.agent_messages:
        if isinstance(message, SystemMessage):
            continue
        role = None
        content = None
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = message.content

        with st.chat_message(role):
            st.markdown(content)

    if user_input := st.chat_input("Ask me anything"):
        st.session_state.agent_messages.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            agent_executor = create_agent_executor(astra_vector_store, st.secrets['OPENAI_API_KEY'], stream_handler)
            response = agent_executor.invoke({"input": user_input, "chat_history": st.session_state.agent_messages})
        st.session_state.agent_messages.append(AIMessage(content=response['output']))
        if len(st.session_state.agent_messages) > 25:
            st.session_state.agent_messages = st.session_state.agent_messages[-25:]


if __name__ == "__main__":
    main()
