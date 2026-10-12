"""
Evaluation metrics for Text-to-SQL.

  - Execution accuracy (EX) — primary metric. Predicted and gold SQL are executed and their
    results compared (multiset comparison, order-sensitive only when the gold query has
    ORDER BY, tolerant to column permutations — following Spider's execution evaluation).
    Rewards semantically equivalent SQL written differently.
  - Exact match (EM) — secondary. sqlglot-normalized comparison that ignores literal values,
    table aliases and identifier case: an approximation of Spider's component-matching EM.
  - Safety-layer rates — valid SQL, hallucinated identifiers, unsafe statements.
  - Hardness — Spider's easy / medium / hard / extra buckets, re-implemented over the
    sqlglot AST (approximation of the official `eval_hardness`).
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from src.safety.sql_validator import HALLUCINATION_CODES, UNSAFE_CODES
from src.schema import DbSchema

HARDNESS_LEVELS = ("easy", "medium", "hard", "extra")
_AGGREGATES = (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min)


# ----------------------------------------------------------------------
# Execution accuracy
# ----------------------------------------------------------------------


def _norm_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        number = float(value)
        return "nan" if math.isnan(number) else round(number, 6)  # 1 == 1.0
    if isinstance(value, bytes):
        return value.hex()
    return value


def _norm_rows(rows) -> list[tuple]:
    return [tuple(_norm_value(v) for v in row) for row in rows]


def results_match(pred_rows, gold_rows, order_matters: bool = False) -> bool:
    """Do two result sets denote the same answer?"""
    pred, gold = _norm_rows(pred_rows), _norm_rows(gold_rows)
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    if len(pred[0]) != len(gold[0]):
        return False

    def same(a: list[tuple], b: list[tuple]) -> bool:
        return a == b if order_matters else Counter(a) == Counter(b)

    if same(pred, gold):
        return True
    n_cols = len(gold[0])
    if n_cols == 1 or n_cols > 6 or (len(gold) > 5000 and n_cols > 3):
        return False
    identity = tuple(range(n_cols))
    for perm in itertools.permutations(identity):  # SELECT b, a vs SELECT a, b
        if perm != identity and same([tuple(r[i] for i in perm) for r in pred], gold):
            return True
    return False


def order_matters(gold_sql: str) -> bool:
    """Spider's convention: results are compared in order iff the gold query sorts."""
    return "order by" in gold_sql.lower()


# ----------------------------------------------------------------------
# Normalized exact match
# ----------------------------------------------------------------------


def normalize_sql(sql: str, schema: DbSchema | None = None, dialect: str = "sqlite") -> str | None:
    """Canonical form: literals masked, aliases resolved, identifiers lower-cased."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except (ParseError, TokenError, ValueError):
        return None
    if tree is None:
        return None

    known = {c.lower() for t in schema.tables for c in t.column_names} if schema else None
    for col in list(tree.find_all(exp.Column)):  # SQLite: unresolvable "x" is a string literal
        ident = col.this
        if (
            known is not None
            and isinstance(ident, exp.Identifier)
            and ident.quoted
            and not col.table
            and ident.name.lower() not in known
        ):
            col.replace(exp.Literal.string(ident.name))

    table_counts = Counter(t.name.lower() for t in tree.find_all(exp.Table))
    alias_to_table = {t.alias.lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.alias}
    for col in tree.find_all(exp.Column):
        qualifier = col.table.lower()
        target = alias_to_table.get(qualifier)
        if target and table_counts[target] == 1:
            col.set("table", exp.to_identifier(target))
    for table in tree.find_all(exp.Table):
        if table.alias and table_counts[table.name.lower()] == 1:
            table.set("alias", None)
    if sum(table_counts.values()) == 1:  # single table: qualifiers carry no information
        for col in tree.find_all(exp.Column):
            col.set("table", None)

    for literal in list(tree.find_all(exp.Literal)):
        literal.replace(exp.Placeholder())
    for ident in tree.find_all(exp.Identifier):
        ident.set("this", ident.this.lower())
        ident.set("quoted", False)
    return tree.sql(dialect=dialect, comments=False)


def exact_match(pred_sql: str | None, gold_sql: str, schema: DbSchema | None = None) -> bool:
    if not pred_sql:
        return False
    pred, gold = normalize_sql(pred_sql, schema), normalize_sql(gold_sql, schema)
    return pred is not None and pred == gold


# ----------------------------------------------------------------------
# Spider hardness (approximation of evaluation.py:eval_hardness)
# ----------------------------------------------------------------------


def _top_level(node: exp.Expression | None, owner: exp.Expression):
    """Nodes of a clause, not descending into nested SELECTs."""
    if node is None:
        return
    for sub in node.walk(prune=lambda n: isinstance(n, exp.Select) and n is not owner):
        if isinstance(sub, exp.Select) and sub is not owner:
            continue
        yield sub


def _condition_units(cond: exp.Expression | None) -> int:
    if cond is None:
        return 0
    if isinstance(cond, exp.And | exp.Or):
        return _condition_units(cond.left) + _condition_units(cond.right)
    if isinstance(cond, exp.Paren):
        return _condition_units(cond.this)
    return 1


def classify_hardness(sql: str, dialect: str = "sqlite") -> str:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except (ParseError, TokenError, ValueError):
        return "extra"
    nested = 0
    while isinstance(tree, exp.SetOperation):  # UNION / INTERSECT / EXCEPT count as nesting
        nested += 1
        tree = tree.left
    if not isinstance(tree, exp.Select):
        return "extra"
    s = tree

    where = s.args.get("where")
    having = s.args.get("having")
    group = s.args.get("group")
    order = s.args.get("order")
    joins = s.args.get("joins") or []
    where_cond = where.this if where else None
    having_cond = having.this if having else None
    join_conds = [j.args.get("on") for j in joins if j.args.get("on") is not None]

    comp1 = sum(1 for clause in (where, group, order, s.args.get("limit")) if clause) + len(joins)
    for cond in [where_cond, having_cond, *join_conds]:
        nodes = list(_top_level(cond, s))
        comp1 += sum(isinstance(n, exp.Or) for n in nodes) + sum(
            isinstance(n, exp.Like) for n in nodes
        )
    for cond in [where_cond, having_cond, *join_conds]:
        nested += sum(
            1 for n in (cond.find_all(exp.Select) if cond is not None else ())
            if n.find_ancestor(exp.Select) is s
        )  # fmt: skip

    aggs = 0
    for part in [*s.expressions, where_cond, group, order, having_cond]:
        aggs += sum(isinstance(n, _AGGREGATES) for n in _top_level(part, s))
    others = (
        int(aggs > 1)
        + int(len(s.expressions) > 1)
        + int(_condition_units(where_cond) > 1)
        + int(bool(group) and len(group.expressions) > 1)
    )

    if comp1 <= 1 and others == 0 and nested == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and nested == 0) or (comp1 <= 2 and others < 2 and nested == 0):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and nested == 0)
        or (2 < comp1 <= 3 and others <= 2 and nested == 0)
        or (comp1 <= 1 and others == 0 and nested <= 1)
    ):
        return "hard"
    return "extra"


# ----------------------------------------------------------------------
# Per-example records and summaries
# ----------------------------------------------------------------------


@dataclass
class EvalRecord:
    idx: int
    db_id: str
    question: str
    gold_sql: str
    mode: str
    pred_sql: str | None
    status: str
    exec_match: bool
    exact_match: bool
    hardness: str
    latency_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int
    violation_codes: list[str] = field(default_factory=list)
    error: str | None = None
    repaired: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _rate(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def summarize(records: list[EvalRecord]) -> dict:
    latencies = [r.latency_ms for r in records]
    summary = {
        "n": len(records),
        "execution_accuracy": _rate([r.exec_match for r in records]),
        "exact_match": _rate([r.exact_match for r in records]),
        "valid_sql_rate": _rate([r.status in ("ok", "exec_error") for r in records]),
        "exec_success_rate": _rate([r.status == "ok" for r in records]),
        "hallucination_rate": _rate([bool(HALLUCINATION_CODES & set(r.violation_codes)) for r in records]),
        "unsafe_generated_rate": _rate([bool(UNSAFE_CODES & set(r.violation_codes)) for r in records]),
        "cannot_answer_rate": _rate([r.status == "cannot_answer" for r in records]),
        "llm_error_rate": _rate([r.status == "llm_error" for r in records]),
        "repaired_rate": _rate([r.repaired for r in records]),
        "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "generation_ms_mean": round(statistics.fmean(r.generation_ms for r in records), 1) if records else 0.0,
        "prompt_tokens_mean": round(statistics.fmean(r.prompt_tokens for r in records), 1) if records else 0.0,
        "completion_tokens_mean": round(statistics.fmean(r.completion_tokens for r in records), 1) if records else 0.0,
        "by_hardness": {},
    }  # fmt: skip
    for level in HARDNESS_LEVELS:
        subset = [r for r in records if r.hardness == level]
        if subset:
            summary["by_hardness"][level] = {
                "n": len(subset),
                "execution_accuracy": _rate([r.exec_match for r in subset]),
            }
    return summary
