"""
Prompt templates for Text-to-SQL generation.

The four modes form a 2x2 experimental design — {base model, QLoRA adapter} x
{full schema, retrieval-augmented context} — so each effect can be measured separately:

  mode        model     context
  ---------   -------   --------------------------------------------------------
  prompt      base      full schema + static few-shot examples (prompt engineering)
  rag         base      BM25-pruned schema + retrieved examples + value hints
  qlora       adapter   full schema (identical to the SFT training format)
  qlora_rag   adapter   BM25-pruned schema + retrieved examples + value hints

Every mode sees the target database schema. Spider is cross-database — evaluation
databases never appear in training — so a fine-tuned model cannot memorise schemas;
SFT teaches it to *read* them and to write consistent SQL.
"""

from __future__ import annotations

PROMPT_VERSION = "v2"  # bump whenever a template changes (part of the cache key)

MODES = ("prompt", "rag", "qlora", "qlora_rag")
RAG_MODES = frozenset({"rag", "qlora_rag"})
ADAPTER_MODES = frozenset({"qlora", "qlora_rag"})
CANNOT_ANSWER = "CANNOT_ANSWER"

# -----------------------------------------------------------------------
# System prompt (same for all modes — sets role and output format)
# -----------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert SQL generator. Given a natural language question and the relevant database schema (CREATE TABLE statements), generate a valid, executable SQLite query.

Rules:
- Output ONLY the SQL query. No explanation, no markdown code fences.
- Use only tables and columns that appear in the provided schema.
- Generate read-only SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, or CREATE.
- If you cannot answer the question from the schema, output: CANNOT_ANSWER
"""

# -----------------------------------------------------------------------
# Static few-shot examples (prompt mode)
# Written on a synthetic schema on purpose: Spider's evaluation databases
# (e.g. concert_singer) must never leak into prompts.
# -----------------------------------------------------------------------
FEW_SHOT_SCHEMA = """CREATE TABLE author (
  author_id NUMBER,
  name TEXT,
  country TEXT,
  birth_year NUMBER,
  PRIMARY KEY (author_id)
);

CREATE TABLE book (
  book_id NUMBER,
  title TEXT,
  author_id NUMBER,
  genre TEXT,
  price NUMBER,
  PRIMARY KEY (book_id),
  FOREIGN KEY (author_id) REFERENCES author(author_id)
);"""

FEW_SHOT_EXAMPLES = [
    {
        "question": "How many authors are from Canada?",
        "sql": "SELECT count(*) FROM author WHERE country = 'Canada'",
    },
    {
        "question": "What is the name of the author who has written the most books?",
        "sql": "SELECT T1.name FROM author AS T1 JOIN book AS T2 ON T1.author_id = T2.author_id "
        "GROUP BY T1.author_id ORDER BY count(*) DESC LIMIT 1",
    },
    {
        "question": "List the titles of books priced above the average price.",
        "sql": "SELECT title FROM book WHERE price > (SELECT avg(price) FROM book)",
    },
    {
        "question": "Which authors have not written any book?",
        "sql": "SELECT name FROM author WHERE author_id NOT IN (SELECT author_id FROM book)",
    },
]


def _schema_block(schema_context: str, value_hints: list[str] | None) -> str:
    block = f"Database Schema:\n{schema_context}"
    if value_hints:
        hints = "\n".join(f"-- {h}" for h in value_hints)
        block += f"\n\n-- Values from the database that the question mentions:\n{hints}"
    return block


def build_prompt(
    question: str,
    schema_context: str,
    mode: str = "rag",
    examples: list[dict] | None = None,
    value_hints: list[str] | None = None,
) -> list[dict]:
    """
    Build a chat-format prompt for the given mode.

    Returns a list of message dicts: [{"role": ..., "content": ...}, ...]
    compatible with vLLM's OpenAI-compatible chat endpoint.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if mode == "prompt":
        # Prompt engineering: static few-shot examples + the full target schema
        few_shot = "\n\n".join(
            f"Question: {ex['question']}\nSQL: {ex['sql']}" for ex in FEW_SHOT_EXAMPLES
        )
        user_content = (
            f"Here are some examples:\n\nSchema:\n{FEW_SHOT_SCHEMA}\n\n{few_shot}\n\n"
            f"Now answer this:\n{_schema_block(schema_context, None)}\n\n"
            f"Question: {question}\nSQL:"
        )

    elif mode in RAG_MODES:
        # Retrieval-augmented: retrieved examples, pruned schema, value hints
        parts = []
        if examples:
            shots = "\n\n".join(
                f"Question: {ex['question']}\nSQL: {ex['query']}" for ex in examples
            )
            parts.append(f"Similar examples (from other databases):\n\n{shots}")
        parts.append(_schema_block(schema_context, value_hints))
        parts.append(f"Question: {question}\nSQL:")
        user_content = "\n\n".join(parts)

    elif mode == "qlora":
        # Fine-tuned adapter: the exact format it was trained on
        user_content = f"{_schema_block(schema_context, None)}\n\nQuestion: {question}\nSQL:"

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: {', '.join(MODES)}")

    messages.append({"role": "user", "content": user_content})
    return messages


def format_sft_example(
    record: dict,
    mode: str = "qlora",
    schema_context: str | None = None,
    examples: list[dict] | None = None,
    value_hints: list[str] | None = None,
) -> dict:
    """
    Convert a processed Spider record into a supervised fine-tuning example.

    Uses TRL's conversational prompt-completion format, so the loss is computed on the
    SQL completion only (not on the schema/question tokens).
    """
    prompt = build_prompt(
        record["question"],
        schema_context if schema_context is not None else record["schema_sql"],
        mode=mode,
        examples=examples,
        value_hints=value_hints,
    )
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": record["query"]}],
        "db_id": record["db_id"],
    }
