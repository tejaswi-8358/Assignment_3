import streamlit as st 
from langchain_together import TogetherEmbeddings
import pandas as pd
import numpy as np
import os 
from pinecone import Pinecone
from together import Together 

df = pd.read_excel('reviews_data.xlsx')

# For the security reasons of uploading my apikeys of together.ai and pine cone and the index host on github in public account 
# i am here not mentioning them , but while running the code i have used them . you can check through my screenshots.

os.environ["TOGETHER_API_KEY"] = "###"

# Initialize the TogetherEmbeddings model
pc = Pinecone(api_key='###')

embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
index = pc.Index(host="###")
client = Together()
#pip install pinecone
#  Create embeddings for all reviews - this will take close to 1.5 to 2 hours
#streamlit
st.title("Hotel Customer Sentiment Analysis")
print("TOGETHER_API_KEY:",os.getenv("TOGETHER_API_KEY"))

query = st.text_input("Enter a query about customer reviews:","How is the food quality?")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
rating_filter = st.slider("Select Rating Filter",1,10,(1,10))

if st.button("Analyze Sentiment"):
    query_embedding = embeddings.embed_query(query)
    
    start_date_str = int(start_date.strftime('%Y%m%d'))
    end_date_str = int(end_date.strftime('%Y%m%d'))
    
    results = index.query(
        vector=query_embedding,
        top_k=10,
        namespace="",
        include_metadata=True,
        filter={
            "Rating": {"$gte": rating_filter[0], "$lte": rating_filter[1]},
            "review_date": {"$gte": start_date_str, "$lte": end_date_str}
        }
    )
    
    matches = results["matches"]
    
    if not matches:
        st.warning("No reviews found matching the criteria.")
    else:
        matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
        
        req_df = df[df["review_id"].isin(matched_ids)]
        concatenated_reviews = " ".join(req_df["Review"].tolist())
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages = [{"role": "user", "content": f"""
                based on the reviews:  {concatenated_reviews}, and query of manager: {query}.
                Stick to the specific query of the manager and keep it short.
                Do not mention the name of the hotel.
            """}]
        )
        
        st.subheader("Sentiment Summary")
        st.write(response.choices[0].message.content)
        
        st.subheader("Matched Reviews")
        st.dataframe(df[["Review","customer_id"]])




