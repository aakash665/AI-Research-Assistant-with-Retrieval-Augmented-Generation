import atexit
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models

"""Loads, indexes, and searches PDF content in Qdrant."""

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
ENV_FILE_PATH = UTILS_DIR / "var.env"
QDRANT_STORAGE_PATH = UTILS_DIR / "qdrant_storage"
INDEXED_DOCUMENTS_PATH = UTILS_DIR / "indexed_documents.json"

load_dotenv(ENV_FILE_PATH)


@lru_cache(maxsize=1)
def get_vector_store_client() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_STORAGE_PATH))


def close_vector_store_client() -> None:
    if get_vector_store_client.cache_info().currsize == 0:
        return

    client = get_vector_store_client()
    try:
        client.close()
    except Exception:
        pass

    try:
        delattr(client, "_client")
    except AttributeError:
        pass

    get_vector_store_client.cache_clear()


atexit.register(close_vector_store_client)


def extract_pdf_title(pdf_path: Path, pages: list[Any]) -> str:
    filename_title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
    if pages:
        metadata_title = " ".join(((pages[0].metadata or {}).get("title") or "").split()).strip()
        if metadata_title and metadata_title.casefold() != filename_title.casefold():
            return metadata_title
        for line in pages[0].page_content.splitlines():
            title_candidate = " ".join(line.split()).strip()
            if len(title_candidate) < 12:
                continue
            if title_candidate.casefold().startswith(("arxiv:", "http://", "https://")):
                continue
            return title_candidate
    return filename_title


def build_document_catalog(pdf_paths: list[Path]) -> tuple[list[Any], list[dict[str, str]]]:
    loaded_documents = []
    document_catalog = []
    for pdf_path in sorted(pdf_paths):
        loaded_pages = PyPDFLoader(str(pdf_path)).load()
        document_title = extract_pdf_title(pdf_path, loaded_pages)
        for page in loaded_pages:
            page.metadata["document_name"] = pdf_path.name
            page.metadata["document_title"] = document_title
        loaded_documents.extend(loaded_pages)
        document_catalog.append({"file_name": pdf_path.name, "title": document_title})
    return loaded_documents, document_catalog


def chunk_documents(docs: list[Any]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=150,
    )
    document_chunks = splitter.split_documents(docs)
    for chunk_index, chunk in enumerate(document_chunks):
        page_number = int(chunk.metadata.get("page", 0)) + 1
        document_name = chunk.metadata.get("document_name", "unknown.pdf")
        chunk.metadata.update(
            {
                "chunk_id": f"chunk_{chunk_index}",
                "page_number": page_number,
                "citation": f"[{document_name} p.{page_number}]",
            }
        )
    return document_chunks


def save_document_catalog(catalog: list[dict[str, str]]) -> None:
    INDEXED_DOCUMENTS_PATH.write_text(
        json.dumps(
            {
                "version": 2,
                "documents": catalog,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_indexed_document_catalog() -> list[dict[str, str]]:
    if INDEXED_DOCUMENTS_PATH.exists():
        try:
            payload = json.loads(INDEXED_DOCUMENTS_PATH.read_text(encoding="utf-8"))
            documents = payload.get("documents", []) if payload.get("version") == 2 else []
            if documents:
                return documents
        except (OSError, json.JSONDecodeError):
            pass

    client = get_vector_store_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    catalog_by_document: dict[str, str] = {}
    scroll_offset = None
    while True:
        stored_points, scroll_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=scroll_offset,
            with_payload=["document_name", "document_title"],
            with_vectors=False,
        )
        for stored_point in stored_points:
            stored_payload = stored_point.payload or {}
            document_name = stored_payload.get("document_name")
            if document_name:
                catalog_by_document.setdefault(
                    document_name,
                    stored_payload.get("document_title") or extract_pdf_title(Path(document_name), []),
                )
        if scroll_offset is None:
            break

    return [
        {"file_name": document_name, "title": title}
        for document_name, title in sorted(catalog_by_document.items())
    ]


COLLECTION_NAME = "document_reports"
OVERSAMPLE_FACTOR = 8
MIN_FETCH_LIMIT = 20

EMBEDDING_MODELS = {
    "small": {"name": "text-embedding-3-small", "size": 1536},
    "large": {"name": "text-embedding-3-large", "size": 3072},
}
EMBEDDING_MODEL_KEY = "small"
EMBEDDING_MODEL_NAME = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["name"]
EMBEDDING_VECTOR_SIZE = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["size"]

embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def create_embedding_points(chunks: list[Any]) -> list[models.PointStruct]:
    chunk_texts = [chunk.page_content for chunk in chunks]
    embedding_vectors = embedding_model.embed_documents(chunk_texts)
    return [
        models.PointStruct(
            id=str(uuid4()),
            vector=embedding_vector,
            payload={
                "content": chunk.page_content,
                "document_name": chunk.metadata.get("document_name", "unknown.pdf"),
                "document_title": chunk.metadata.get("document_title", ""),
                "page_number": chunk.metadata.get("page_number", 0),
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "citation": chunk.metadata.get("citation", ""),
                "source": chunk.metadata.get("source", ""),
            },
        )
        for chunk, embedding_vector in zip(chunks, embedding_vectors)
    ]


def recreate_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


def index_documents(pdf_dir: Path) -> dict:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    loaded_documents, document_catalog = build_document_catalog(pdf_paths)
    if not loaded_documents:
        raise ValueError(f"Unable to load PDFs from {pdf_dir}")

    document_chunks = chunk_documents(loaded_documents)
    client = get_vector_store_client()
    recreate_collection(client)

    embedding_points = create_embedding_points(document_chunks)
    for batch_start in range(0, len(embedding_points), 128):
        embedding_batch = embedding_points[batch_start:batch_start + 128]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=embedding_batch,
            wait=True,
        )

    save_document_catalog(document_catalog)

    return {
        "num_pdfs": len(pdf_paths),
        "num_chunks": len(document_chunks),
        "collection_name": COLLECTION_NAME,
    }


def search_similar_documents(
    query: str,
    per_doc_topk: int = 3,
    max_results: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    client = get_vector_store_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    query_embedding = embedding_model.embed_query(query)
    fetch_limit = max(max_results or 0, per_doc_topk * OVERSAMPLE_FACTOR, MIN_FETCH_LIMIT)
    search_points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=fetch_limit,
        with_payload=True,
    ).points

    ranked_results: list[dict[str, Any]] = []
    result_counts_by_document: dict[str, int] = {}

    for search_point in search_points:
        relevance_score = float(search_point.score or 0.0)
        if score_threshold is not None and relevance_score < score_threshold:
            continue

        result_payload = search_point.payload or {}
        document_name = result_payload.get("document_name", "unknown.pdf")
        current_count = result_counts_by_document.get(document_name, 0)
        if current_count >= per_doc_topk:
            continue

        result_counts_by_document[document_name] = current_count + 1
        ranked_results.append(
            {
                "document_name": document_name,
                "document_title": result_payload.get("document_title", ""),
                "page_number": result_payload.get("page_number", 0),
                "chunk_id": result_payload.get("chunk_id", ""),
                "citation": result_payload.get("citation", ""),
                "content": result_payload.get("content", ""),
                "score": relevance_score,
            }
        )

    ranked_results.sort(key=lambda item: item["score"], reverse=True)
    return ranked_results[:max_results] if max_results else ranked_results
