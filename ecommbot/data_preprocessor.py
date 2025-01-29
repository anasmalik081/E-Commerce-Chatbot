import pandas as pd
from langchain_core.documents import Document

def data_preprocessor(path=r"D:\Ecommerce Chatbot\data\flipkart_product_review.csv"):
    df = pd.read_csv(path)
    data = df[["product_title", "review"]]

    product_list = []
    # Iterate over the rows of the DataFrame
    for idx, row in data.iterrows():
        # contruct the object with "product_name" and "reveiew" attributes
        obj = {
            "product_name": row["product_title"],
            "review": row["review"]
        }
        # Append the object to the list
        product_list.append(obj)

    docs = []
    for entry in product_list:
        metadata = {"product_name": entry["product_name"]}
        doc = Document(page_content=entry["review"], metadata=metadata)
        docs.append(doc)

    return docs