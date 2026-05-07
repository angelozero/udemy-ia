import os

from dotenv import load_dotenv
from factory_service import get_chat_model, get_embeddings
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = get_embeddings()
llm = get_chat_model()

pinecone_vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings
)

retriever = pinecone_vectorstore.as_retriever(searh_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
        - Answer the question based only on the following context:
        
        { context }
        
        - Question: { question }
        
        Provide a detailer answer:
    """)


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_chain_without_llm(query: str):
    pass

if __name__ == "__main__":
    print("OKS")
    
    query = "What Vectors embeddings does?"
    
    