import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import textwrap
# import streamlit as st

### Keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

### Establish Pinecone Client and Connection
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('document-embeddings')

### Establish OpenAi Client
openai.api_key = openai_api_key
client = openai.OpenAI()

### Get embeddings
def get_embeddings(text, model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

## Get context
def get_contexts(query, embed_model, k):
    query_embeddings = get_embeddings(query, model=embed_model)
    pinecone_response = index.query(vector=query_embeddings, top_k=k, include_metadata=True)
    context = [item['metadata']['text'] for item in pinecone_response['matches']]
    source = [item['metadata']['page_name'] for item in pinecone_response['matches']]
    # print(source)
    return context, source, query

### Augmented Prompt
def augmented_query(user_query, embed_model='text-embedding-ada-002', k=3):
    context, source, query = get_contexts(user_query, embed_model=embed_model, k=k)
    return "\n\n---\n\n".join(context) + "\n\n---\n\n" + query, source

### Ask GPT
def ask_gpt(system_prompt, user_prompt, model, temp=0.7):
    temperature_ = temp
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature_,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    lines = (completion.choices[0].message.content).split("\n")
    lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
    return "\n".join("\n".join(list) for list in lists)

### Education ChatBot
def Education_ChatBot(query):
    embed_model = 'text-embedding-ada-002'
    primer = """
    You are an educational assistant specializing in computer science, public health, and geography.
   

    You offer:
    - Clear and concise answers to real-time questions.
    - Summarization of discussions to help teachers review key points.
    - Create Quizes

    If the information needed to answer a question is not available, respond with: 
    'I'm sorry, I don't know.'
    If asked who created you, respond with:
    I was created by Team Atom AI. The team members are Josephat Oyondi, Elie Niringiyimana, Jean Habyarimana.
    
    If asked to provide a list, respond using a list format, with each item on a new line, preceded by a bullet point.

    The maximum amount of words for each response should be 300.

    When creating a summary, start with the phrase 'Here is a summary of your question:' followed by a concise response that captures the main points.
    
    If asked 'Thank you', respond with:
    'I'm happy you found the answer useful! If you have any more questions, feel free to ask below. Enjoy your day!'
    Your tone is helpful, supportive, and focused on creating an interactive and effective learning environment.
    """

    llm_model = 'gpt-3.5-turbo'
    user_prompt, source = augmented_query(query, embed_model)
    return ask_gpt(primer, user_prompt, model=llm_model), source


# ### Streamlit Interface ###
# st.title("Education ChatBot")

# # Get user input
# user_input = st.text_input("Ask a question:", "")

# if st.button("Submit"):
#     if user_input:
#         response = Education_ChatBot(user_input)
#         st.write("Response:")
#         st.write(response)
#     else:
#         st.write("Please enter a question or request.")
########################################################################################
#  Always identify the relevant subject area and provide responses based exclusively on your knowledge base, which covers: 
#     Computer Science (algorithms, operating systems, networks, AI/ML, databases, software engineering, programming, cybersecurity, cloud computing, HCI), 
#     Public Health (mental health in USA schools/prisons, human/environmental health, biology, biostatistics, infectious disease control), 
#     and Geography (physical/human geography, GIS, climate change, sustainable development, economic/urban/political/cultural geography). 
#     Structure responses with subject identification, clear answers using the knowledge base, examples when possible, and optional follow-up questions. 
#     When requested, generate comprehensive quizzes containing 10 multiple-choice questions and 10 short-answer questions relevant to the topic discussed. 
#     Use clear, educational language and encourage critical thinking. 
#     If information isn't in your knowledge base, respond with "I don't have enough information to answer this question accurately." 
#     Your purpose is to enhance student engagement, facilitate real-time Q&A, support interactive learning, gauge understanding through 
#     quizzes and discussions, and identify areas needing additional support, all while staying within your knowledge boundaries.
########################################################################################
