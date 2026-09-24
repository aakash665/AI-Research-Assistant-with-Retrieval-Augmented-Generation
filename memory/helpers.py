import json
from typing import Any

def infer_route_used(
    document_evidence_chunks: list[Any],
    web_search_results: list[Any],
) -> str:
    if document_evidence_chunks and web_search_results:
        return "both"
    if document_evidence_chunks:
        return "documents"
    if web_search_results:
        return "web"
    return "none"


def build_evidence_context(
    serialized_evidence: str,
    *,
    include_formatted_context: bool = False,
) -> dict[str, Any]:
    if not serialized_evidence:
        return {"has_evidence": False, "summary": "None", "formatted_evidence": ""}

    try:
        evidence_payload = json.loads(serialized_evidence)
    except json.JSONDecodeError:
        return {
            "has_evidence": False,
            "summary": "Active evidence is available, but it could not be summarized.",
            "formatted_evidence": serialized_evidence,
        }

    document_evidence = evidence_payload.get("document_evidence") or {}
    web_evidence = evidence_payload.get("web_evidence") or {}
    document_evidence_chunks = document_evidence.get("chunks") or []
    web_search_results = web_evidence.get("results") or []
    retrieval_route = evidence_payload.get("route_used") or infer_route_used(
        document_evidence_chunks,
        web_search_results,
    )
    evidence_summary = evidence_payload.get("summary") or "None"
    contains_evidence = bool(document_evidence_chunks or web_search_results)
    context_summary = "\n".join(
        [
            f"Route used: {retrieval_route}",
            f"Summary: {evidence_summary}",
            f"Document chunk count: {len(document_evidence_chunks)}",
            f"Web result count: {len(web_search_results)}",
        ]
    )

    if not include_formatted_context or not contains_evidence:
        return {
            "has_evidence": contains_evidence,
            "summary": context_summary,
            "formatted_evidence": "",
        }

    formatted_evidence_sections = [
        f"Retrieval summary:\nRoute used: {retrieval_route}\nSummary: {evidence_summary}",
        f"Document evidence:\n{json.dumps(document_evidence, indent=2)}" if document_evidence_chunks else "",
        f"Web evidence:\n{json.dumps(web_evidence, indent=2)}" if web_search_results else "",
    ]
    formatted_context = "\n\n".join(
        section for section in formatted_evidence_sections if section
    )

    return {
        "has_evidence": contains_evidence,
        "summary": context_summary,
        "formatted_evidence": formatted_context,
    }