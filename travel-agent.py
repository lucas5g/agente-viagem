from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables.base import RunnableSequence 
from langchain_core.prompts import PromptTemplate
import bs4
import os
import streamlit as st


llm = ChatOpenAI(model="gpt-3.5-turbo")

# query = "Vou viajar para Londres em agosto de 2024. Quero que faça um roteiro de viagem para mim com os eventos que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Londres."


def research_agent(query):
    tools = load_tools(["ddg-search", "wikipedia"], llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, prompt=prompt,
    )

    res = agent_executor.invoke({"input":query})['output']

    return res

def load_data():
    loader = WebBaseLoader(
        web_path="https://www.dicasdeviagem.com/inglaterra/",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading"))
        ),
    )

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

    return vector_store.as_retriever()


def load_data():
    loader = WebBaseLoader(
        web_path="https://www.dicasdeviagem.com/inglaterra/",
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading"))
        ),
    )

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

    return vector_store.as_retriever()

def get_relevant_docs(query: str):
    retriever = load_data()
    relevant_documents = retriever.invoke(query)
    return relevant_documents


def supervisor_agent(query, web, relevant_documents):
    prompt_template = """
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completa e detalhada.
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes 
    para elaborar o roteiro de viagens   
    Contexto: {web}
    Documento relevante: {relevant_documents}
    Usuário: {query},
    Assitente:
    """

    prompt = PromptTemplate(
        input_variables=["web", "relevant_documents", "query"], template=prompt_template
    )

    sequence = RunnableSequence(prompt | llm)
    
    return sequence.invoke({"web": web, "relevant_documents": relevant_documents, "query": query})

def get_response(query):
    web = research_agent(query)
    relevant_documents = get_relevant_docs(query)
    return supervisor_agent(query, web, relevant_documents).content


def lambda_handler(event, context):
    query = event.get('question')
    
    response = get_response(query).content 
    return {"body": response, "status":200}
    
    
st.title("Agente de Viagens")
st.write("Saiba tudo sobre a Inglaterra.")

query = st.chat_input('Busque informações sobre viagens para inglaterra')

if query:
    response = get_response(query)
    st.write(response)
    