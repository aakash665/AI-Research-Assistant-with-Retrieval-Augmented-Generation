from pathlib import Path
import sqlite3
from .helpers import build_evidence_context

"""Stores session queries and cached retrieval evidence."""

UTILITIES_DIR = Path(__file__).resolve().parents[1] / "utils"
MEMORY_DATABASE_PATH = UTILITIES_DIR / "memory.db"


def get_memory_connection() -> sqlite3.Connection:
    database_connection = sqlite3.connect(MEMORY_DATABASE_PATH)
    database_connection.row_factory = sqlite3.Row
    return database_connection


def init_memory() -> None:
    with get_memory_connection() as database_connection:
        database_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_memory (
                session_id TEXT PRIMARY KEY,
                last_user_query TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        database_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_memory (
                session_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_last_user_query(session_id: str, latest_user_query: str) -> None:
    with get_memory_connection() as database_connection:
        database_connection.execute(
            """
            INSERT INTO session_memory (session_id, last_user_query)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_user_query = excluded.last_user_query,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, latest_user_query),
        )


def save_evidence(
    session_id: str,
    evidence_query: str,
    serialized_evidence: str,
) -> None:
    with get_memory_connection() as database_connection:
        database_connection.execute(
            """
            INSERT INTO evidence_memory (session_id, query, evidence_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                query = excluded.query,
                evidence_json = excluded.evidence_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (session_id, evidence_query, serialized_evidence),
        )


def get_session_context(session_id: str) -> dict[str, str]:
    with get_memory_connection() as database_connection:
        session_record = database_connection.execute(
            """
            SELECT last_user_query
            FROM session_memory
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        evidence_record = database_connection.execute(
            """
            SELECT query, evidence_json
            FROM evidence_memory
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    session_context = {
        "last_user_query": session_record["last_user_query"] if session_record else "",
        "cached_query": "",
        "cached_evidence_json": "",
        "cached_evidence_summary": "None",
    }
    if not evidence_record:
        return session_context

    serialized_evidence = evidence_record["evidence_json"]
    evidence_context_data = build_evidence_context(serialized_evidence)
    if not evidence_context_data["has_evidence"]:
        return session_context

    return {
        **session_context,
        "cached_query": evidence_record["query"],
        "cached_evidence_json": serialized_evidence,
        "cached_evidence_summary": evidence_context_data["summary"],
    }