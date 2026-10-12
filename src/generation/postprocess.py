"""Extract the SQL statement from raw LLM output."""

from __future__ import annotations

import re

from src.generation.prompts import CANNOT_ANSWER

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_PREFIX = re.compile(r"^\s*(?:sql\s*query|sqlquery|sql|answer|query)\s*:\s*", re.IGNORECASE)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)
_PROSE_LINE = re.compile(
    r"^\s*(explanation|note|this query|the query|here is|here's|in this query)\b", re.IGNORECASE
)


def extract_sql(text: str) -> str:
    """Best-effort extraction of a single SQL statement from a model response.

    Handles markdown fences, "SQL:" prefixes and trailing explanations. Deliberately does
    NOT split on ';' — a second statement (e.g. an injected DROP TABLE) must stay visible
    so the validator rejects it and the metric records it.
    """
    if not text:
        return ""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    candidate = _PREFIX.sub("", candidate.strip())

    start = _SQL_START.search(candidate)
    if start is None:
        return CANNOT_ANSWER if CANNOT_ANSWER in text else candidate.strip()
    candidate = candidate[start.start() :]

    lines = []
    for line in candidate.splitlines():
        if not line.strip() or _PROSE_LINE.match(line):
            break  # a blank line or prose marks the end of the query
        lines.append(line.rstrip())
    sql = "\n".join(lines).strip().rstrip("`").strip()
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql
