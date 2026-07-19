"""
LearningAssistant - QA over your own document using LangChain + Ollama (local, free).

Requires Ollama running locally (https://ollama.com) with these models pulled:
    ollama pull llama3.2
    ollama pull nomic-embed-text

Usage:
    python main.py

It will:
1. Load text from document.txt
2. Split it into chunks
3. Embed the chunks locally and store them in an in-memory vector store
4. Let you ask questions in a loop, answered using only that document
"""

import os

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT_PATH = "document.txt"
CHAT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

SYSTEM_PROMPT = (
    "Use the given context to answer the question. "
    "If the answer isn't in the context, say you don't know. "
    "Keep the answer concise.\n\n"
    "Context:\n{context}"
)


def load_document(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find '{path}'. Put the text you want to ask questions about in there."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_answer_fn(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_texts(chunks)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm

    def answer(question: str) -> str:
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)
        response = chain.invoke({"context": context, "question": question})
        return response.content

    return answer


def main():
    print(f"Loading document from {DOCUMENT_PATH} ...")
    text = load_document(DOCUMENT_PATH)

    print("Building embeddings + vector store (calling local Ollama) ...")
    try:
        answer = build_answer_fn(text)
    except Exception as e:
        print(
            "\nCouldn't reach Ollama. Make sure:\n"
            "  1. Ollama is installed and running (https://ollama.com)\n"
            "  2. You've pulled the models: 'ollama pull llama3.2' and "
            "'ollama pull nomic-embed-text'\n"
        )
        raise

    print("\nReady! Ask questions about the document. Type 'exit' or 'quit' to stop.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        print(f"Answer: {answer(query)}\n")


if __name__ == "__main__":
    main()
