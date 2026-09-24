from .model_runner import run_model

WRITER_MODEL = "gpt-5.4"
WRITER_REASONING_EFFORT = "low"

"""Writes a grounded report from retrieved evidence."""
def writer(user_query: str, evidence_text: str, verbose: bool = False) -> str:
    if verbose:
        print("[Writer] Writing report...")

    instructions = (
        """
        Answer using only the evidence. Start with the answer.
        Be clear and concise. Include key facts, comparisons, and caveats.
        State supported conclusions clearly. Keep follow-ups brief.
        Do not add unsupported facts. Use exact PDF citations.
        Cite web sources as [Exact Title](Exact URL).
        Say when evidence is weak or incomplete. End after the answer.
        """
    )

    input_text = (
        f"User query: {user_query}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        model=WRITER_MODEL,
        reasoning_effort=WRITER_REASONING_EFFORT,
        tools=None,
    )
    return response.output_text
