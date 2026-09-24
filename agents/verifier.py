from .model_runner import run_model

VERIFIER_MODEL = "gpt-5.4"
VERIFIER_REASONING_EFFORT = "low"

"""Checks a report against evidence and returns the final answer."""
def verifier(
    user_query: str,
    written_draft: str,
    evidence_text: str,
    verbose: bool = False,
) -> str:
    if verbose:
        print("[Verifier] Checking report...")

    instructions = (
        """
        Check the draft against the evidence and query. Return only the final answer.
        Start with a direct answer. Remove irrelevant or unsupported claims.
        Keep useful facts, comparisons, and caveats. Keep follow-ups brief.
        Add exact PDF citations and web links as [Exact Title](Exact URL).
        Say when evidence is weak or incomplete. End after the last cited sentence.
        """
    )

    input_text = (
        f"User query: {user_query}\n\n"
        f"Report draft:\n{written_draft}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        model=VERIFIER_MODEL,
        reasoning_effort=VERIFIER_REASONING_EFFORT,
        tools=None,
    )
    return response.output_text
