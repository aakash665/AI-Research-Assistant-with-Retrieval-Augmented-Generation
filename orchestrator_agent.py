from datetime import datetime
import json
from typing import Any, Dict
from memory import (
    build_evidence_context,
    get_session_context,
    save_evidence,
    save_last_user_query,
)
from agents import (
    retriever,
    run_model,
    verifier,
    writer,
)
"""Coordinates retrieval, writing, verification, and cached evidence reuse."""

ORCHESTRATOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "reuse_cached_evidence",
                "description": "Reuse evidence for a related follow-up, not a new topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why it applies.",
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_retriever",
            "description": "Retrieve evidence from PDFs or the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clear retrieval query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_writer",
            "description": "Write from the active evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "User request.",
                },
                "evidence_text": {
                    "type": "string",
                    "description": "Evidence for the draft.",
                },
            },
            "required": ["user_query", "evidence_text"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_verifier",
            "description": "Check the draft and return the final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "written_draft": {
                    "type": "string",
                    "description": "Draft to verify.",
                },
                "evidence_text": {
                    "type": "string",
                    "description": "Evidence to check.",
                },
            },
            "required": ["written_draft", "evidence_text"],
            "additionalProperties": False,
        },
    },
]

ORCHESTRATOR_INSTRUCTIONS = (
    f"""
    You coordinate retrieval, writing, and verification.

    Classify each turn:
    1. social:
    greetings, thanks, brief replies, or goodbyes.

    2. out_of_scope:
    unrelated requests.

    3. evidence_request:
    factual questions needing PDF or web evidence.

    Rules:
    - For social turns, reply in one short sentence. Use no tools.
    - For out_of_scope turns, use no tools and reply: "I focus on document and web research."
    - Treat named topics, products, papers, companies, and current events as evidence requests.
    - Use call_retriever only for evidence requests.
    - Choose tools from the current state.
    - Retrieve, write, then verify.
    - Return the verified answer when available.
    """
)
ORCHESTRATOR_MODEL = "gpt-5.4-mini"
ORCHESTRATOR_REASONING_EFFORT = "low"

def build_orchestrator_prompt_context(state: dict[str, Any]) -> str:
    evidence_context = build_evidence_context(state["evidence_json"])
    return (
        f"- current_date: {state['current_date']}\n"
        f"- last_user_query: {state['last_user_query'] or 'None'}\n"
        f"- cached_query: {state['cached_query'] or 'None'}\n"
        f"- retrieval_attempted: {state['retrieval_attempted']}\n"
        f"- has_written_draft: {bool(state['written_draft'])}\n"
        f"- has_verification: {bool(state['verification'])}\n\n"
        f"Cached evidence summary:\n{state['cached_evidence_summary']}\n\n"
        f"Current evidence summary:\n{evidence_context['summary']}"
    )

def orchestrator_agent(
    user_query: str,
    session_id: str = "default",
    verbose: bool = True,
) -> Dict[str, Any]:
    
    context = get_session_context(session_id)
    cached_evidence = context["cached_evidence_json"]

    today = datetime.now().astimezone().date().isoformat()

    state = {
        "user_query": user_query,
        "current_date": today,
        "last_user_query": context["last_user_query"],
        "cached_query": context["cached_query"],
        "cached_evidence_summary": context["cached_evidence_summary"],
        "retrieval_attempted": False,
        "evidence_json": "",
        "written_draft": "",
        "verification": "",
        "final_answer": "",
    }
    active_message = "Evidence is active; proceed to writing."

    response = run_model(
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=ORCHESTRATOR_MODEL,
        reasoning_effort=ORCHESTRATOR_REASONING_EFFORT,
        input_data=(
            f"User query: {user_query}\n\n"
            f"Current state:\n{build_orchestrator_prompt_context(state)}"
        ),
        tools=ORCHESTRATOR_TOOL_SCHEMAS,
    )

    for _ in range(4):
        calls = [item for item in response.output if item.type == "function_call"]

        if not calls:
            state["final_answer"] = response.output_text or "I could not find enough evidence."
            save_last_user_query(session_id, user_query)
            return state

        results = []

        for call in calls:
            name = call.name
            args = json.loads(call.arguments)

            if name == "reuse_cached_evidence":
                if state["evidence_json"]:
                    output = active_message
                elif cached_evidence:
                    if verbose:
                        print("[Orchestrator] Reusing evidence.")
                    state["evidence_json"] = cached_evidence
                    state["cached_query"] = context["cached_query"]
                    output = f"Reused cached evidence: {args['reason']}"
                else:
                    output = "No cached evidence is available."

            elif name == "call_retriever":
                if state["evidence_json"]:
                    output = active_message
                elif state["retrieval_attempted"]:
                    output = "Retrieval was already attempted."
                else:
                    state["retrieval_attempted"] = True
                    if verbose:
                        print("[Orchestrator] Starting retriever...")
                    evidence = retriever(
                        args["query"].strip() or state["user_query"],
                        last_user_query=state["last_user_query"],
                        verbose=verbose,
                    )
                    output = evidence.model_dump_json()
                    evidence_context = build_evidence_context(output)
                    if evidence_context["has_evidence"]:
                        state["evidence_json"] = output
                        state["cached_query"] = state["user_query"]
                        save_evidence(session_id, state["user_query"], output)
                        state["cached_evidence_summary"] = evidence_context["summary"]
                    else:
                        output = evidence.summary or "Not enough evidence was found."

            elif name in {"call_writer", "call_verifier"}:
                evidence_context = build_evidence_context(
                    state["evidence_json"],
                    include_formatted_context=True,
                )
                formatted_evidence = evidence_context["formatted_evidence"]
                if name == "call_writer":
                    if not formatted_evidence:
                        output = "Cannot write without evidence."
                    else:
                        if verbose:
                            print("[Orchestrator] Starting writer...")
                        output = writer(
                            user_query=state["user_query"],
                            evidence_text=formatted_evidence,
                            verbose=verbose,
                        )
                        state["written_draft"] = output
                elif not state["written_draft"]:
                    output = "Cannot verify without a draft."
                elif not formatted_evidence:
                    output = "Cannot verify without evidence."
                else:
                    if verbose:
                        print("[Orchestrator] Starting verifier...")
                    output = verifier(
                        user_query=state["user_query"],
                        written_draft=state["written_draft"],
                        evidence_text=formatted_evidence,
                        verbose=verbose,
                    )
                    state["verification"] = output

            else:
                output = "Unknown tool."

            results.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

        response = run_model(
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model=ORCHESTRATOR_MODEL,
            reasoning_effort=ORCHESTRATOR_REASONING_EFFORT,
            input_data=[
                *results,
                {
                    "type": "message",
                    "role": "user",
                    "content": f"Updated state:\n{build_orchestrator_prompt_context(state)}",
                },
            ],
            tools=ORCHESTRATOR_TOOL_SCHEMAS,
            previous_response_id=response.id,
        )

    if state["verification"]:
        state["final_answer"] = state["verification"]
        save_last_user_query(session_id, user_query)
    else:
        state["final_answer"] = "Stopped after the maximum steps."

    if verbose:
        print("[Orchestrator] Done.")
    return state
