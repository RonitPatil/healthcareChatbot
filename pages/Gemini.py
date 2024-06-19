import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from utils.data_loader import populate_vector_store, scrape_link
from utils.initialize_vector_store import initialize_vector_store
from utils.create_chains import create_conversational_retrieval_chain

from trulens_eval import TruChain, OpenAI, Tru
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval import Feedback
import numpy as np


if 'tru_initialized' not in st.session_state:
    tru = Tru()
    tru.reset_database()
    st.session_state['tru_initialized'] = True
else:
    tru = st.session_state['tru']

st.session_state['tru'] = tru

if 'google_chat_history' not in st.session_state:
    st.session_state['google_chat_history'] = []
if "google_messages" not in st.session_state:
    st.session_state.google_messages = []

prompt_template = """You are very powerful assistant that can answer questions about diseases and also diagnose users \
based on their symptoms based on the provided context. ALWAYS! display the source link along with the content. You can \
also greet people when they greet you. If you are asked questions about anything other than healthcare, you need to \
politely inform the user that you are a healthcare assistant and can only provide information on healthcare-related \
topics. If you do not get relevant documents from the tool, politely inform the user that you could not find relevant \
information in your database.\
CONTEXT: {context}
Question: {question}
Helpful Answer:"""


def main():
    astra_vector_store = initialize_vector_store(st.secrets['ASTRA_DB_APPLICATION_TOKEN'], st.secrets['ASTRA_DB_ID'])
    st.set_page_config(page_title="Healthcare Chatbot", page_icon=":robot_face:")
    st.header('Gemini Healthcare Chatbot')
    with st.sidebar:
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
            if st.button("Submit Link", key="submit_link"):
                try:
                    scrape_link(new_link, astra_vector_store)
                    st.success("Link successfully scraped and processed!")
                except Exception as e:
                    st.error(f"Failed to scrape link: {e}")

    for message in st.session_state.google_messages:
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

    conversational_retrieval_chain_with_google = create_conversational_retrieval_chain(astra_vector_store,
                                                                                       st.secrets['OPENAI_API_KEY'],
                                                                                       st.secrets['GOOGLE_API_KEY'],
                                                                                       st.secrets['ANTHROPIC_API_KEY'],
                                                                                       model='Google')

    # Initialize provider class
    openai = OpenAI()

    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval.app import App
    context = App.select_context(conversational_retrieval_chain_with_google)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(openai.relevance, name="Relevance between Q/A").on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(openai.context_relevance, name="Relevance between Q and Context")
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    class OpenAI_custom(fOpenAI):
        def no_answer_feedback(self, question: str, response: str) -> float:
            return float(self.endpoint.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system",
                     "content": "Does the RESPONSE provide an answer to the QUESTION? Rate on a scale of 1 to 10. \
                     Respond with the number only."},
                    {"role": "user", "content": f"QUESTION: {question}; RESPONSE: {response}"}
                ]
            ).choices[0].message.content) / 10

        def answer_feedback(self, question: str, response: str) -> float:
            return float(self.endpoint.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system",
                     "content": "How factually correct is the RESPONSE to the QUESTION? Rate on a scale of 1 to 10. \
                     Respond with the number only."},
                    {"role": "user", "content": f"QUESTION: {question}; RESPONSE: {response}"}
                ]
            ).choices[0].message.content) / 10

    custom = OpenAI_custom()

    # No answer feedback (custom)
    f_no_answer = Feedback(custom.no_answer_feedback, name="Accuracy between Q/A").on_input_output()
    f_answer = Feedback(custom.no_answer_feedback, name="Groundedness").on_input_output()

    google_conversational_retrieval_chain_recorder = TruChain(
        conversational_retrieval_chain_with_google,
        app_id="Conversation-Retrieval-Chain-feedback-Google",
        feedbacks=[f_qa_relevance, f_context_relevance, f_no_answer, f_answer]
    )

    if user_input := st.chat_input("Ask me anything"):
        st.session_state.google_messages.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        with google_conversational_retrieval_chain_recorder as recording:
            response_data = conversational_retrieval_chain_with_google({
                "question": user_input,
                "chat_history": st.session_state["google_chat_history"]
            })
            print(response_data)
            st.session_state["google_chat_history"].append((user_input, response_data["answer"]))
            st.chat_message("assistant").markdown(response_data["answer"])
            st.session_state.google_messages.append(AIMessage(content=response_data["answer"]))
            if len(st.session_state.google_messages) > 25:
                st.session_state.google_messages = st.session_state.google_messages[-25:]


if __name__ == "__main__":
    tru.run_dashboard(port=8505)
    main()
