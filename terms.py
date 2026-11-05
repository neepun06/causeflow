"""Bridge between an AI extraction and the cash-flow engine, via a human-verified table."""
from __future__ import annotations

from engine import ClauseTerms
from extract import CLAUSE_SPECS

LD_UNIT_TO_MONTH = {"per_day": 30.0, "per_week": 30.0 / 7.0, "per_month": 1.0}

# key, clause, field, default when absent/unstated, neutral when unverified, unit, label
PARAMS = [
    ("advance_pct", "mobilisation_advance", "percent", 0, 0, "% of contract", "Advance amount"),
    ("advance_interest_pct", "mobilisation_advance", "interest_rate_pct", 0, 0, "% p.a.", "Interest on advance"),
    ("advance_recovery_start_pct", "mobilisation_advance", "recovery_start_pct", 10, 10, "% progress", "Recovery starts at"),
    ("advance_recovery_end_pct", "mobilisation_advance", "recovery_end_pct", 80, 80, "% progress", "Fully recovered by"),
    ("retention_pct", "retention", "percent_per_bill", 0, 0, "% of each bill", "Retention per bill"),
    ("retention_cap_pct", "retention", "cap_pct_of_contract", 0, 0, "% of contract (0 = no cap)", "Retention cap"),
    ("retention_release_completion_pct", "retention", "release_at_completion_pct", 0, 0, "% of retention", "Released at completion"),
    ("billing_cycle_months", "billing_cycle", "months", 1, 1, "months", "Billing cycle"),
    ("payment_days", "payment_period", "days", 30, 0, "days", "Payment period"),
    ("ld_rate_pct_month", "liquidated_damages", None, 0, 0, "% of contract per month", "LD rate"),
    ("ld_cap_pct", "liquidated_damages", "cap_pct", 0, 0, "% of contract", "LD cap"),
    ("pv_coverage_pct", "price_variation", None, 0, 100, "% of escalation recovered", "Price variation coverage"),
    ("pbg_pct", "performance_bank_guarantee", "percent", 0, 0, "% of contract", "Performance BG"),
    ("dlp_months", "defect_liability", "months", 0, 0, "months", "Defect liability period"),
]


def _num(x):
    try:
        return float(x) if x is not None and x != "" else None
    except (TypeError, ValueError):
        return None


def ld_rate_per_month(values: dict) -> float | None:
    rate = _num(values.get("rate_pct"))
    if rate is None:
        return None
    unit = str(values.get("rate_unit") or "per_month").lower().replace(" ", "_")
    return rate * LD_UNIT_TO_MONTH.get(unit, 1.0)


def primary_value(clause_key: str, cl: dict):
    """The single value we score for each clause in validation."""
    if not cl or not cl.get("found"):
        return None
    v = cl.get("values") or {}
    if clause_key == "liquidated_damages":
        return ld_rate_per_month(v)
    if clause_key == "price_variation":
        a = v.get("applicable")
        return a if isinstance(a, bool) else str(a).lower() == "true"
    return _num(v.get(CLAUSE_SPECS[clause_key]["primary"]))


def extraction_to_rows(extraction: dict) -> list[dict]:
    rows = []
    clauses = extraction.get("clauses", {})
    for key, ck, field, default, _neutral, unit, label in PARAMS:
        cl = clauses.get(ck) or {}
        v = cl.get("values") or {}
        if ck == "liquidated_damages" and field is None:
            val = ld_rate_per_month(v)
        elif ck == "price_variation":
            app = v.get("applicable")
            app = app if isinstance(app, bool) else str(app).lower() == "true"
            cov = _num(v.get("coverage_pct"))
            val = (cov if cov is not None else 100.0) if (cl.get("found") and app) else 0.0
        else:
            val = _num(v.get(field))
        if not cl.get("found"):
            source = "not in contract (AI)" if extraction.get("model") != "manual" else "not in contract"
            val = default if val is None else val
        elif val is None:
            source, val = "assumed default (not stated)", default
        else:
            ref = cl.get("clause_ref") or ""
            source = f"p.{cl.get('page')} {ref}".strip()
        check = (cl.get("check") or {}).get("status", "manual entry" if extraction.get("model") == "manual" else "")
        rows.append(dict(
            param=key, clause=CLAUSE_SPECS[ck]["label"], term=label, value=float(val), unit=unit,
            source=source, citation_check=check, quote=cl.get("quote") or "", verified=False,
        ))
    return rows


def rows_to_terms(rows: list[dict]) -> tuple[ClauseTerms, list[str]]:
    """Verified rows go into the forecast; unverified rows fall back to their neutral value."""
    neutral = {p[0]: p[4] for p in PARAMS}
    vals, unverified = {}, []
    for r in rows:
        if r.get("verified"):
            vals[r["param"]] = r["value"]
        else:
            vals[r["param"]] = neutral[r["param"]]
            unverified.append(r["term"])
    for k in ("billing_cycle_months", "dlp_months"):
        vals[k] = int(round(vals[k]))
    return ClauseTerms(**vals), unverified
