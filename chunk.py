import json
import os
import shutil
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


JSON_FILE = "final_knowledge.json"
DB_DIR = "./db"
EMBED_MODEL = "embeddinggemma"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 80


def load_json_data(json_file: str) -> List[Dict]:
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("JSON data is empty or invalid.")

    return data


def convert_to_documents(data: List[Dict]) -> List[Document]:
    docs: List[Document] = []

    for item in data:
        heading = item.get("heading", "No Heading")
        title = item.get("title", heading)
        content = item.get("content", "").strip()
        page = item.get("page", "unknown")
        section = item.get("section", "unknown")
        source = item.get("source", "")
        category = item.get("category", "general")

        if not content:
            continue

        text = f"Title: {title}\nHeading: {heading}\nContent: {content}"

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "page": page,
                    "section": section,
                    "heading": heading,
                    "source": source,
                    "category": category,
                },
            )
        )

    if not docs:
        raise ValueError("No valid documents were created from the JSON input.")

    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def db_exists(db_dir: str) -> bool:
    return os.path.exists(db_dir) and any(os.scandir(db_dir))


def build_or_load_vectorstore(split_docs: List[Document], force_rebuild: bool = False) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if force_rebuild and os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    if db_exists(DB_DIR) and not force_rebuild:
        print("[OK] Loading existing vector DB...")
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

    print("[OK] Building new vector DB...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    return vectorstore


def main(force_rebuild: bool = True) -> None:
    data = load_json_data(JSON_FILE)
    print(f"[OK] Loaded records: {len(data)}")

    docs = convert_to_documents(data)
    print(f"[OK] Documents created: {len(docs)}")

    split_docs = split_documents(docs)
    print(f"[OK] Chunks created: {len(split_docs)}")

    build_or_load_vectorstore(split_docs, force_rebuild=force_rebuild)
    print("[OK] Vector DB ready.")


if __name__ == "__main__":
    main(force_rebuild=True)
