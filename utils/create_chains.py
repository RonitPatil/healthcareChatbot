from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import CTransformers
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

MEMORY_KEY = "chat_history"


def create_agent_executor(astra_vector_store, openai_api_key, stream_handler):
    retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        "data_search",
        "Search for information about diseases in the vector database and return the most relevant \
        information."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are very powerful assistant with access to tools that can help you retrieve \
            data from the vector database, answer questions about diseases and also diagnose users based on their \
            symptoms. You can also greet people when they greet you. If you are asked questions about anything other \
            than healthcare, you need to politely inform the user that you are a healthcare assistant and can only \
            provide information on healthcare-related topics. If you do not get relevant documents from the tool, \
            politely inform the user that you could not find relevant information in your database."),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    chat_llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        openai_api_key=openai_api_key,
        temperature=0.5,
        streaming=True,
        callbacks=[stream_handler],
    )

    tools = [retriever_tool]
    llm_with_tools = chat_llm.bind_tools(tools)

    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x[MEMORY_KEY],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

    return agent_executor


def create_retriever_chain(astra_vector_store, openai_api_key):
    openai_chat_llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        openai_api_key=openai_api_key,
        temperature=0.5,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are very powerful assistant that can answer questions about diseases and also diagnose users \
        based on their symptoms based on the provided context. Display the source link along with the content. You can \
        also greet people when they greet you. If you are asked questions about anything other than healthcare, you \
        need to politely inform the user that you are a healthcare assistant and can only provide information on \
        healthcare-related topics. If you do not get relevant documents from the tool, politely inform the user that \
        you could not find relevant information in your database.\
        CONTEXT: {context}"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=openai_chat_llm,
        prompt=prompt,
    )

    retriever = astra_vector_store.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain


def create_history_aware_retriever_chain(astra_vector_store, openai_api_key):
    openai_chat_llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        openai_api_key=openai_api_key,
        temperature=0.5,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are very powerful assistant that can answer questions about diseases and also diagnose users \
        based on their symptoms based on the provided context. Display the source link along with the content. You can \
        also greet people when they greet you. If you are asked questions about anything other than healthcare, you \
        need to politely inform the user that you are a healthcare assistant and can only provide information on \
        healthcare-related topics. If you do not get relevant documents from the tool, politely inform the user that \
        you could not find relevant information in your database.\
        CONTEXT: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=openai_chat_llm,
        prompt=prompt,
    )

    retriever = astra_vector_store.as_retriever()

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human",
         "Given the above conversation, generate a search query to look up in order to get information relevant to "
         "the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=openai_chat_llm,
        retriever=retriever,
        prompt=retriever_prompt,
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain,
    )

    return retrieval_chain


def create_conversational_retrieval_chain(astra_vector_store,
                                          openai_api_key,
                                          google_api_key,
                                          claude_api_key,
                                          model='OpenAI'):
    openai_chat_llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        openai_api_key=openai_api_key,
        temperature=0.5,
    )
    google_chat_llm = ChatGoogleGenerativeAI(
        google_api_key=google_api_key,
        model="gemini-pro",
        temperature=0.5,
        convert_system_message_to_human=True,
    )

    claude_chat_llm = ChatAnthropic(
        temperature=0.5,
        api_key=claude_api_key,
        model_name="claude-3-opus-20240229"
    )

    if model == 'OpenAI':
        chat_llm = openai_chat_llm
    elif model == 'Google':
        chat_llm = google_chat_llm
    else:
        chat_llm = claude_chat_llm

    chain = ConversationalRetrievalChain.from_llm(
        chat_llm,
        retriever=astra_vector_store.as_retriever(),
        chain_type="stuff",
        verbose=True,
        max_tokens_limit=150,
        return_source_documents=True,
    )
    return chain
