"""
Ask questions using the local RAG pipeline.

Run:

    python query.py
"""

from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K,
)

from utils import (
    print_documents,
    get_embeddings,
    get_llm,
    format_documents,
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

PROMPT = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the provided context.

If the answer is not present in the context,
say you don't know.

Context:

{context}

Question:

{question}
"""
)


# ---------------------------------------------------------
# Load Chroma
# ---------------------------------------------------------

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR),
    embedding_function=get_embeddings(),
)


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)


# ---------------------------------------------------------
# Debug Retriever
# ---------------------------------------------------------

def retrieve(question: str):

    documents = retriever.invoke(question)

    print_documents(documents)

    return format_documents(documents)


# ---------------------------------------------------------
# Chain
# ---------------------------------------------------------

chain = (

    {

        "context": retrieve,

        "question": RunnablePassthrough(),

    }

    | PROMPT

    | get_llm()

    | StrOutputParser()

)


# ---------------------------------------------------------
# Loop
# ---------------------------------------------------------

print()

print("=" * 80)

print("Local RAG")

print("=" * 80)

print()

while True:

    question = input("\nQuestion > ").strip()

    if question.lower() in {

        "exit",

        "quit",

        "q",

    }:

        break

    print()

    answer = chain.invoke(question)

    print()

    print("=" * 80)

    print(answer)

    print("=" * 80)