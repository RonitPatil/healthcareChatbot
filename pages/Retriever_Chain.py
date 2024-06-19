import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler


from utils.data_loader import populate_vector_store, scrape_link
from utils.initialize_vector_store import initialize_vector_store
from utils.create_chains import create_retriever_chain


def update_usage(cb: OpenAICallbackHandler) -> None:
    callback_properties = [
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_cost",
    ]
    for prop in callback_properties:
        value = getattr(cb, prop, 0)
        st.session_state.usage[prop] += value


def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "usage" not in st.session_state:
        st.session_state.usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
        }

    astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])

    st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
    st.header('Retriever Healthcare Chatbot')
    with st.sidebar:
        st.header("API Usage")
        if st.session_state["usage"]:
            st.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
            st.metric("Total Costs in $", round(st.session_state["usage"]["total_cost"], 2))
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

    for message in st.session_state.chat_history:
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
    retriever_chain = create_retriever_chain(astra_vector_store,
                                             st.secrets['OPENAI_API_KEY'])
    if user_input := st.chat_input("Ask me anything"):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        with get_openai_callback() as cb:
            response = retriever_chain.invoke({"input": user_input})
            update_usage(cb)
        st.chat_message("assistant").markdown(response['answer'])
        st.session_state.chat_history.append(AIMessage(content=response['answer']))
        if len(st.session_state.chat_history) > 25:
            st.session_state.chat_history = st.session_state.chat_history[-25:]


if __name__ == "__main__":
    main()
