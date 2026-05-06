import os
from dotenv import load_dotenv
from factory_service import get_chat_model

load_dotenv()

def main():
    llm = get_chat_model()
    response = llm.invoke("Diga apenas ola")
    print(response.content)

if __name__ == '__main__':
    main()