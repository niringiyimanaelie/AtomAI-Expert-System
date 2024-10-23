import json
import openai
import torch
import pinecone
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from pinecone import Pinecone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os
import sys
# import streamlit as st

### Keys
load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone and OpenAI clients
pc = Pinecone(api_key=pinecone_api_key)
index_name = "document-embeddings"

openai.api_key = openai_api_key
client = openai.OpenAI()

index = pc.Index(index_name)

def load_extracted_text(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    extracted_data = [(entry['extracted_text'], entry['page_name']) for entry in data if 'extracted_text' in entry]
    return extracted_data

def encode_documents(documents):
    embeddings = []
    for doc in documents:
        response = client.embeddings.create(input=doc[0], model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)  # Corrected line
    return embeddings



def insert_into_pinecone(embeddings, documents, batch_size=100):
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        
        data = [
            (f"doc_{i+j}", embedding, {"page_name": doc[1], "text": doc[0]})
            for j, (embedding, doc) in enumerate(zip(batch_embeddings, batch_documents))
        ]
        
        # Check the size of the data
        data_size = sys.getsizeof(data)
        print(f"Batch {i//batch_size + 1}: Size = {data_size} bytes")
        
        if data_size > 4194304:  # 4 MB limit
            print("Warning: This batch exceeds Pinecone's size limit.")
        
        index.upsert(vectors=data)


def generate_answer(query, generator):
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    query_embedding = [float(x) for x in response.data[0].embedding]
    
    # Print the embedding to debug
    print("Query embedding:", query_embedding)
    print("Query embedding length:", len(query_embedding))  # Should be 1536

    # Check for NaN or infinite values
    if np.any(np.isnan(query_embedding)) or np.any(np.isinf(query_embedding)):
        raise ValueError("Embedding contains NaN or infinite values.")

    # Verify the length
    if len(query_embedding) != 1536:
        raise ValueError(f"Unexpected embedding dimension: {len(query_embedding)}. Expected: 1536.")
    
    query_results = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    retrieved_docs = [result['metadata'] for result in query_results['matches']]
    context = "\n".join([f"Page: {doc['page_name']}, Text: {doc['text']}" for doc in retrieved_docs])
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return answer, context

def main(json_file_path, query):
    documents = load_extracted_text(json_file_path)
    embeddings = encode_documents(documents)
    insert_into_pinecone(embeddings, documents)
    generator = pipeline("text-generation", model="gpt2")
    answer, context = generate_answer(query, generator)
    return answer, context

if __name__ == "__main__":
    json_file_path = 'json/ocr_results_geo.json'
    query = "What is human geography?"
    answer, context = main(json_file_path, query)
    print("Answer:", answer)
    print("Context:", context)
