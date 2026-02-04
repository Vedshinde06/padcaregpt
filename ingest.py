from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = Path("padcare_data")
VECTOR_DB_DIR = Path("vectorstore")


def load_documents(data_dir: Path):
    documents = []
    for file_path in data_dir.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(str(VECTOR_DB_DIR))


def main():
    print("Loading documents...")
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    create_vector_store(chunks)

    print("Ingestion completed successfully.")
    print(f"Vector store saved at: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
