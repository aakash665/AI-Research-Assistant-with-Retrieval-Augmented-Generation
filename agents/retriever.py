import json
import os
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel
from memory import infer_route_used
from vector_db import load_indexed_document_catalog, search_similar_documents
from .model_runner import run_model
from tavily import TavilyClient

"""Retrieves evidence from indexed PDFs and the web."""

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) 

class ResearchEvidencePack(BaseModel):
    query: str
    route_used: Literal["documents", "web", "both", "none"]
    summary: str
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None


def retrieve_document(
    query: str,
    per_doc_topk: int = 4,
    score_threshold: Optional[float] = 0.2,
) -> Dict[str, Any]:
    try:
        results = search_similar_documents(
            query=query,
            per_doc_topk=per_doc_topk,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        return {
            "query": query,
            "summary": f"Document retrieval failed: {type(exc).__name__}",
            "chunks": [],
        }
    return {
        "query": query,
        "summary": (
            "Retrieved relevant evidence from the uploaded PDFs."
            if results else
            "No sufficiently relevant evidence was found in the uploaded PDFs."
        ),
        "chunks": [
            {
                "document_name": item["document_name"],
                "document_title": item.get("document_title") or item["document_name"],
                "page_number": int(item["page_number"]),
                "chunk_id": item["chunk_id"],
                "citation": item["citation"],
                "content": item["content"],
                "score": float(item["score"]),
            }
            for item in results
        ],
    }

def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    if tavily is None:
        return {"query": query, "results": []}

    try:
        result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=num_results,
            include_answer=False,
            include_raw_content=True,
            include_images=False,
        )
        return {"query": query, "results": result.get("results", [])}
    except Exception:
        return {"query": query, "results": []}


RETRIEVER_MODEL = "gpt-5.4-mini"
RETRIEVER_REASONING_EFFORT = "low"

RETRIEVER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "retrieve_document",
        "description": (
            """Search indexed PDFs and return cited evidence. Use for PDF questions."""
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rewrite the request as a clear PDF search query."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            """Search the web for current or non-PDF information with cited sources."""
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rewrite the request as a clear web search query."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

RETRIEVER_INSTRUCTIONS = """
You retrieve evidence for a research assistant.

Indexed document titles are provided in the input.

Rules:
- Use a retrieval tool for valid evidence requests.
- Use document titles to choose PDF or web search.
- Use web_search when PDFs are unrelated or empty.
- Use both only when needed.
- Use the last query only for follow-ups.
- Return a short summary, not the final answer.
"""

def retriever(
    user_query: str,
    *,
    last_user_query: str = "",
    verbose: bool = False,
) -> ResearchEvidencePack:
    
    previous_response_id: Optional[str] = None

    document_catalog = load_indexed_document_catalog()
    indexed_documents_text = "\n".join(
        f"- {item['title']} (file: {item['file_name']})"
        for item in document_catalog
    ) or "- None"
    pending_input: Any = (
        f"Current user query: {user_query.strip()}\n"
        f"Last user query: {last_user_query.strip() or 'None'}\n"
        f"Indexed document titles and topic hints:\n{indexed_documents_text}"
    )
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None
    summary = ""

    for _ in range(4):
        response = run_model(
            instructions=RETRIEVER_INSTRUCTIONS,
            input_data=pending_input,
            model=RETRIEVER_MODEL,
            tools=RETRIEVER_TOOL_SCHEMAS,
            previous_response_id=previous_response_id,
            reasoning_effort=RETRIEVER_REASONING_EFFORT,
        )
        previous_response_id = response.id

        tool_results = []
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            summary = (response.output_text or "").strip()
            break

        for call in function_calls:
            query = json.loads(call.arguments)["query"].strip()

            if call.name == "retrieve_document":
                if verbose:
                    print("[Retriever] Searching documents...")
                function_response = retrieve_document(query)
                if function_response.get("chunks"):
                    document_evidence = function_response
            else:
                if verbose:
                    print("[Retriever] Searching the web...")
                function_response = web_search(query)
                if function_response.get("results"):
                    web_evidence = function_response

            tool_results.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(function_response),
                }
            )

        pending_input = tool_results

    document_chunks = document_evidence.get("chunks") if document_evidence else []
    web_results = web_evidence.get("results") if web_evidence else []
    route_used = infer_route_used(document_chunks or [], web_results or [])

    return ResearchEvidencePack(
        query=user_query,
        route_used=route_used,
        summary=summary,
        document_evidence=document_evidence if document_evidence and document_evidence.get("chunks") else None,
        web_evidence=web_evidence if web_evidence and web_evidence.get("results") else None,
    )