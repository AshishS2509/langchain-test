
from langchain_ollama import ChatOllama


def main():
    propmt = "write a hello world program in python."

    chat = ChatOllama(temperature=0, model="lfm2.5-thinking:latest")

    print(chat.invoke(input=propmt).content)


if __name__ == "__main__":
    main()
