import os
import pandas as pd
from ecommbot.data_preprocessor import data_preprocessor
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Loading environments
load_dotenv()

ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")

# Loading embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Defining the ingestion
def ingestdata(status):
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="ecommbot",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE
    )

    storge = status
    if storge is None:
        docs = data_preprocessor()
        inserted_ids = vector_store.add_documents(docs)
    else:
        return vector_store
    return vector_store, inserted_ids

if __name__ == "__main__":
    vector_store, inserted_ids = ingestdata(None)
    print(f"\nInderted {len(inserted_ids)} documents.")
    results = vector_store.similarity_search("Can you tell me the low budget sound basshead.")
    for res in results:
        print(f"{res.page_content} - [{res.metadata}]")