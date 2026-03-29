import os
from dotenv import load_dotenv

load_dotenv()
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SOURCE_DOCUMENTS = ["sourceFiles/Bjarne Stroustrup - A Tour of C++-Addison-Wesley (2013).pdf"]
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    # index all the given source files
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    # generate embedding for the 
    db = generate_embed_index(all_docs)


def ingest_docs(source_documents):
    all_docs = []
    for source_doc in source_documents:
        print(source_doc)
        docs = pdf_to_chunks(source_doc)
        all_docs = all_docs + docs
    return all_docs


def pdf_to_chunks(pdf_file):
    # Use the tokenizer from the embedding model to determine the chunk size
    # so that chunks don't get truncated.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n \n", "\n\n", "\n", " ", ""],
        chunk_size=512,
        chunk_overlap=0,
    )
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split(text_splitter)
    return docs


def generate_embed_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    if chroma_persist_dir:
        db = create_index_chroma(docs, embeddings, chroma_persist_dir)
    else:
        raise EnvironmentError("No vector store environment variables found.")
    return db


def create_index_chroma(docs, embeddings, persist_dir):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    db.persist()
    return db

if __name__ == "__main__":
    main()