from __future__ import annotations

from uuid import uuid4
from pathlib import Path

from memory import init_memory
from orchestrator_agent import orchestrator_agent
from vector_db import close_vector_store_client, index_documents

def initialize_app(docs_dir: Path) -> None:
    init_memory()
    print("Ingesting documents...")
    index_info = index_documents(docs_dir)
    print(
        f"{index_info['num_pdfs']} PDFs ingested to Qdrant collection {index_info['collection_name']}"
    )

def chat_with_supervisor(session_id: str | None = None) -> None:
    if session_id is None:
        session_id = str(uuid4())

    print("Analyze your pdfs!! \n")
    print("Use 'q', 'exit', or 'exist' to end chat. \n")
    while True:
        query = input("User: ").strip()

        if not query:
            continue

        if query.lower() in {"q", "exit", "exist"}:
            print("Exiting chat loop.")
            break

        answer = orchestrator_agent(query, session_id=session_id, verbose=True)
        print("Assistant:", answer["final_answer"])
        print()

if __name__ == "__main__":
    docs_dir = Path("docs")
    try:
        initialize_app(docs_dir)
        chat_with_supervisor()
    finally:
        close_vector_store_client()
