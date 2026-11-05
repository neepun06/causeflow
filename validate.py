"""Score an AI extraction against a manual (human) extraction: correct / missed / incorrect."""
from __future__ import annotations

import pandas as pd

from extract import CLAUSE_SPECS
from terms import LD_UNIT_TO_MONTH, primary_value

TEMPLATE_COLUMNS = ["clause_key", "found", "value", "unit", "page", "clause_ref", "notes"]

TEMPLATE_HELP = {
    "mobilisation_advance": "advance as % of contract value",
    "retention": "% deducted from each gross bill",
    "billing_cycle": "months between bills (1 = monthly)",
    "payment_period": "calendar days to pay a bill (working days x 7/5)",
    "liquidated_damages": "rate as % of contract value; put per_day / per_week / per_month in unit",
    "price_variation": "TRUE if a price variation clause applies, FALSE if absent/deleted",
    "performance_bank_guarantee": "% of contract value",
    "defect_liability": "months",
}


def template() -> pd.DataFrame:
    return pd.DataFrame([
        dict(clause_key=k, found="", value="", unit="", page="", clause_ref="", notes=TEMPLATE_HELP[k])
        for k in CLAUSE_SPECS
    ], columns=TEMPLATE_COLUMNS)


def _truthy(x) -> bool:
    return str(x).strip().lower() in {"yes", "y", "true", "1", "found"}


def _gt_value(row) -> object:
    key = row["clause_key"]
    if key == "price_variation":
        return _truthy(row["value"])
    try:
        v = float(row["value"])
    except (TypeError, ValueError):
        return None
    if key == "liquidated_damages":
        unit = str(row.get("unit") or "per_month").strip().lower().replace(" ", "_")
        v *= LD_UNIT_TO_MONTH.get(unit, 1.0)
    return v


def _match(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if a is None or b is None:
        return False
    return abs(a - b) <= max(0.01, 0.01 * abs(b))


def score(extraction: dict, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    truth = truth.copy()
    truth["clause_key"] = truth["clause_key"].astype(str).str.strip()
    rows = []
    for key, spec in CLAUSE_SPECS.items():
        t = truth[truth["clause_key"] == key]
        if t.empty:
            continue
        t = t.iloc[0]
        cl = extraction["clauses"].get(key, {})
        ai_found, gt_found = bool(cl.get("found")), _truthy(t["found"])
        ai_val, gt_val = primary_value(key, cl), (_gt_value(t) if gt_found else None)
        if not gt_found and not ai_found:
            outcome = "correct"
            detail = "absent, and AI said absent"
        elif not gt_found and ai_found:
            outcome = "incorrect"
            detail = "AI reported a clause the contract does not have (invented)"
        elif gt_found and not ai_found:
            outcome = "missed"
            detail = "clause is in the contract but AI did not find it"
        elif _match(ai_val, gt_val):
            outcome = "correct"
            detail = "value matches"
        else:
            outcome = "incorrect"
            detail = "wrong value"
        try:
            page_ok = int(float(t["page"])) == int(cl.get("page")) if gt_found and ai_found else None
        except (TypeError, ValueError):
            page_ok = None
        rows.append(dict(
            clause=spec["label"], manual_value=gt_val, ai_value=ai_val, outcome=outcome, detail=detail,
            manual_page=t["page"], ai_page=cl.get("page"), same_page=page_ok,
            citation_check=(cl.get("check") or {}).get("status", ""),
        ))
    df = pd.DataFrame(rows)
    counts = df["outcome"].value_counts().to_dict() if not df.empty else {}
    summary = dict(
        total=len(df), correct=counts.get("correct", 0), missed=counts.get("missed", 0),
        incorrect=counts.get("incorrect", 0),
        accuracy=(counts.get("correct", 0) / len(df)) if len(df) else 0.0,
    )
    return df, summary
