from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings
import cassio


def initialize_vector_store(astra_db_application_token, astra_db_id):
    embedding = OpenAIEmbeddings()
    cassio.init(token=astra_db_application_token, database_id=astra_db_id)
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="general",
        session=None,
        keyspace=None,
    )
    return astra_vector_store
