import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from ecommbot.ingestion import ingestdata

# Setting the API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ")

# Define generation
def generation(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
        Your ecommercebot bit is an expert in product recommendations and customer queries.
        It analyzes product titles and reviews to provide accurate and helpful responses.
        Ensure your answers are relevant to the product context and refrain from straying off-topic.
        Your responses should be concise and informative.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER: 
    """
    prompt = ChatPromptTemplate.from_template(template=PRODUCT_BOT_TEMPLATE)
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":    
    vector_store = ingestdata("done")
    chain = generation(vector_store)
    print(chain.invoke("Can you tell me the low budget sound bass headset."))
    print(chain.invoke("Can you tell me the best bluetooth buds."))