import os

from dotenv import load_dotenv
from factory_service import get_chat_model, get_embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS

load_dotenv()


def main():
    def format_docs(docs):
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)

    query = "What Vectors embeddings does?"

    llm = get_chat_model()
    embeddings = get_embeddings()

    pinecone_vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )

    retriever = pinecone_vectorstore.as_retriever(searh_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = format_docs(docs)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Responda usando exclusivamente o conteudo fornecido."),
            ("human", "{question}\n\nContexto: {context}\n\nResposta: "),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": query, "context": context})
    print(response)


if __name__ == "__main__":
    main()
