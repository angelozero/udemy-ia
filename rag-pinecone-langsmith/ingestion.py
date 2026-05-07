import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from factory_service import get_chat_model, get_embeddings

load_dotenv()


def main():

    loader = TextLoader("rag-pinecone-langsmith/data/data.txt")
    document = loader.load()
    llm = get_chat_model()
    embeddings = get_embeddings()
    pinecone_index_name = os.getenv("INDEX_NAME")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    
    PineconeVectorStore.from_documents(texts, embeddings, index_name=pinecone_index_name)
    
    # response = llm.invoke("Diga apenas ola")

    # print(response.content)

    print("Fim")

if __name__ == "__main__":
    main()
