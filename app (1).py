"""ClauseFlow: AI contract analyser for financial risk clauses (CE725 course project, Topic 12)."""
from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from engine import ProjectInputs, clause_impacts, combined_cost, forecast, linear_curve, s_curve, sensitivity
from extract import CLAUSE_SPECS, DEFAULT_MODEL, MODEL_CHOICES, extract, list_models, recheck
from terms import extraction_to_rows, rows_to_terms
from validate import score, template

import sys
sys.path.insert(0, str(Path(__file__).parent))
HERE = Path(__file__).parent
SAMPLES = HERE  # saved extractions (*.json) sit next to app.py

st.set_page_config(page_title="ClauseFlow", page_icon="📑", layout="wide")


def lakh(x: float) -> str:
    return f"₹{x:,.1f} L"


def secret_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


def load_extraction(ex: dict, pdf: bytes | None = None):
    st.session_state.extraction = ex
    st.session_state.pdf = pdf
    st.session_state.rows_base = extraction_to_rows(ex)
    st.session_state.rows = st.session_state.rows_base
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 📑 ClauseFlow")
    st.caption("Reads a construction contract's financial clauses, then shows what they do to the contractor's cash.")
    server_key = secret_key()  # stays on the server; never sent to the browser
    typed_key = st.text_input("Gemini API key" + (" (optional)" if server_key else ""), value="", type="password",
                              help="Free key from aistudio.google.com. Needed only for 'Upload a contract PDF'.")
    api_key = typed_key or server_key
    if server_key and not typed_key:
        st.caption("Using the app's built-in key.")
    model = st.selectbox("Model", MODEL_CHOICES, index=0, accept_new_options=True,
                         help="Type another model id if Google renames these.")
    if api_key and st.button("Check which models my key can use", width="stretch"):
        try:
            st.write(", ".join(list_models(api_key)))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not list models: {e}")
    st.divider()
    st.caption("**Privacy.** On Gemini's free tier Google may use what you send to improve its models. "
               "Upload public tender documents only. For confidential contracts use a paid key or a local model.")
    st.caption("CE725 Construction Economics & Finance · IIT Bombay · Topic 12")

st.title("ClauseFlow")
st.markdown("**AI contract analyser for financial risk clauses.** Extract → verify → forecast cash flow → price the risk.")

tab_ex, tab_ver, tab_cf, tab_sens, tab_val, tab_about = st.tabs(
    ["① Extract", "② Verify", "③ Cash flow", "④ Sensitivity", "⑤ Validation", "Method & limits"])

# ------------------------------------------------------------------ 1. Extract
with tab_ex:
    mode = st.segmented_control("Source", ["Upload a contract PDF", "Load a saved extraction", "Demo terms (no AI)"],
                                default="Upload a contract PDF")
    if mode == "Upload a contract PDF":
        pdf = st.file_uploader("Tender or contract PDF (text-based, not scanned)", type=["pdf"])
        if st.button("Extract clauses", type="primary", disabled=pdf is None):
            if not api_key:
                st.error("Add a Gemini API key in the sidebar, or use 'Load a saved extraction'.")
            else:
                with st.spinner("Reading the contract and extracting 8 clause types…"):
                    try:
                        data = pdf.getvalue()
                        load_extraction(extract(data, pdf.name, api_key, model or DEFAULT_MODEL), data)
                        st.success("Done. Check the citation column, then verify each term in step ②.")
                    except Exception as e:  # noqa: BLE001
                        msg = str(e)
                        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                            msg = "Free-tier rate limit hit. Wait a minute and retry, or switch to a Flash-Lite model."
                        st.error(f"Extraction failed: {msg}")
    elif mode == "Load a saved extraction":
        saved = sorted(p.name for p in SAMPLES.glob("*.json"))
        choice = st.selectbox("Saved extractions", saved or ["(none yet)"])
        up_json = st.file_uploader("…or upload an extraction JSON you downloaded earlier", type=["json"])
        up_pdf = st.file_uploader("Optional: the same contract PDF, to re-run the citation check", type=["pdf"],
                                  key="recheck_pdf")
        if st.button("Load", type="primary"):
            if not up_json and choice == "(none yet)":
                st.error("No saved extractions yet. Upload a JSON file.")
                st.stop()
            ex = json.loads(up_json.getvalue()) if up_json else json.loads((SAMPLES / choice).read_text())
            pdf_bytes = up_pdf.getvalue() if up_pdf else None
            if pdf_bytes and ex.get("model") != "manual":
                ex = recheck(ex, pdf_bytes)
            load_extraction(ex, pdf_bytes)
            st.success(f"Loaded: {ex.get('source')}")
    else:
        if st.button("Load demo terms", type="primary"):
            load_extraction(json.loads((HERE / "demo_cpwd_typical_terms.json").read_text()))
            st.success("Loaded typical CPWD-conditions terms, entered by hand. Verify them in step ②.")

    ex = st.session_state.get("extraction")
    if ex:
        st.subheader(ex.get("source", "Extraction"))
        clauses = ex["clauses"]
        found = sum(1 for c in clauses.values() if c.get("found"))
        grounded = sum(1 for c in clauses.values() if (c.get("check") or {}).get("status", "").startswith(("verbatim", "near")))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clauses found", f"{found} / {len(CLAUSE_SPECS)}")
        c2.metric("Quotes found on cited page", f"{grounded} / {found}" if ex.get("model") != "manual" else "n/a")
        c3.metric("Pages read", f"{len(ex.get('pages_sent') or [])} of {ex.get('n_pages') or '–'}")
        c4.metric("Model", ex.get("model", "–"))
        table = []
        for key, cl in clauses.items():
            vals = cl.get("values") or {}
            table.append(dict(
                Clause=CLAUSE_SPECS[key]["label"], Found="✅" if cl.get("found") else "—",
                Values=", ".join(f"{k}={v}" for k, v in vals.items() if v not in (None, "")),
                Page=cl.get("page"), Ref=cl.get("clause_ref"),
                Citation=(cl.get("check") or {}).get("status", ""), Confidence=cl.get("confidence"),
                Quote=cl.get("quote"), Note=cl.get("note"),
            ))
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", column_config={
            "Quote": st.column_config.TextColumn(width="large"), "Values": st.column_config.TextColumn(width="medium")})
        st.caption("Citation check: ClauseFlow searches the PDF text for each quote. 'not found in document' "
                   "means the AI may have paraphrased or invented the clause, so read that page yourself.")
        st.download_button("Download this extraction (JSON)", json.dumps(ex, indent=2),
                           file_name=f"extraction_{Path(ex.get('source', 'contract')).stem}.json", mime="application/json")

# ------------------------------------------------------------------ 2. Verify
with tab_ver:
    if "rows" not in st.session_state:
        st.info("Run or load an extraction in step ① first.")
        terms, unverified = rows_to_terms(extraction_to_rows(json.loads((HERE / "demo_cpwd_typical_terms.json").read_text())))
        unverified = ["all"]
    else:
        st.markdown("Check each term against the contract page shown, correct the value if needed, then tick **Verified**. "
                    "**Only verified terms reach the forecast**; unticked terms are held at a neutral value.")
        edited = st.data_editor(
            pd.DataFrame(st.session_state.rows_base), hide_index=True, width="stretch",
            key=f"verify_editor_{st.session_state.get('editor_version', 0)}",
            disabled=["param", "clause", "term", "unit", "source", "citation_check", "quote"],
            column_order=["verified", "clause", "term", "value", "unit", "source", "citation_check", "quote"],
            column_config={
                "verified": st.column_config.CheckboxColumn("Verified", width="small"),
                "value": st.column_config.NumberColumn("Value", format="%.2f"),
                "quote": st.column_config.TextColumn("Quote (from AI)", width="large"),
                "citation_check": st.column_config.TextColumn("Citation check"),
            })
        st.session_state.rows = edited.to_dict("records")
        terms, unverified = rows_to_terms(st.session_state.rows)
        n_ok = len(st.session_state.rows) - len(unverified)
        if unverified:
            st.warning(f"{n_ok} of {len(st.session_state.rows)} terms verified. Held at neutral: {', '.join(unverified)}.")
        else:
            st.success("All terms verified. The forecast uses exactly these values.")
    st.session_state.terms, st.session_state.unverified = terms, unverified

# ------------------------------------------------------------------ 3. Cash flow
with tab_cf:
    st.markdown("#### Project")
    a, b, c, d = st.columns(4)
    V = a.number_input("Contract value (₹ lakh)", 10.0, 1_000_000.0, 1000.0, step=50.0)
    N = b.number_input("Duration (months)", 3, 60, 12)
    curve = c.selectbox("Work schedule", ["S-curve", "Linear", "Custom"])
    cost = d.number_input("Contractor cost (% of work value)", 50.0, 120.0, 88.0, step=1.0,
                          help="Direct + indirect cost before financing. 88% means a 12% margin.")
    a, b, c, d = st.columns(4)
    od = a.number_input("Overdraft rate (% p.a.)", 0.0, 30.0, 11.0, step=0.25)
    disc = b.number_input("Cost of capital for NPV (% p.a.)", 0.0, 30.0, 12.0, step=0.25)
    infl = c.number_input("Input cost inflation (% p.a.)", 0.0, 20.0, 6.0, step=0.5)
    delay = d.number_input("Delay scenario for LD (months late)", 0.0, 24.0, 0.0, step=0.5)
    with st.expander("Bank guarantee costs and supplier credit"):
        a, b, c = st.columns(3)
        bgc = a.number_input("BG commission (% p.a.)", 0.0, 5.0, 1.5, step=0.25)
        bgm = b.number_input("Cash margin on BG (% of BG)", 0.0, 100.0, 25.0, step=5.0)
        lagc = c.number_input("Supplier credit (months)", 0, 3, 0)

    if curve == "S-curve":
        sched = s_curve(int(N))
    elif curve == "Linear":
        sched = linear_curve(int(N))
    else:
        base = pd.DataFrame({"month": range(1, int(N) + 1), "work_pct": s_curve(int(N))})
        sched = st.data_editor(base, hide_index=True, disabled=["month"], key=f"sched_{N}")["work_pct"].tolist()

    p = ProjectInputs(contract_value=V, duration_months=int(N), schedule_pct=sched, cost_ratio_pct=cost,
                      cost_lag_months=int(lagc), od_rate_pct=od, discount_rate_pct=disc, inflation_pct=infl,
                      delay_months=delay, bg_commission_pct=bgc, bg_margin_pct=bgm)
    terms = st.session_state.terms
    if st.session_state.unverified == ["all"]:
        st.info("No verified terms yet: showing typical CPWD terms for illustration. Complete step ② for your contract.")
        from engine import ClauseTerms
        terms = ClauseTerms()
    elif st.session_state.unverified:
        st.warning("Provisional: some terms are unverified and held at neutral values (see step ②).")
    try:
        df, m = forecast(p, terms)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    imp = clause_impacts(p, terms)
    combo = combined_cost(p, terms)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Peak funding requirement", lakh(m["peak_funding"]),
              help="Largest overdraft the contractor needs at any point, including interest.")
    k2.metric("Peak in month", m["peak_month"] if m["peak_month"] is not None else "never negative")
    k3.metric("Overdraft interest", lakh(m["od_interest"]))
    k4.metric(f"NPV @ {disc:g}%", lakh(m["npv"]), help="Present value of all project cash flows, before overdraft interest.")
    k5.metric("Profit after interest", lakh(m["profit_after_interest"]),
              delta=f"{m['profit_after_interest'] / V * 100:.1f}% of contract", delta_color="off")

    bal = alt.Chart(df).mark_area(opacity=0.25, line=True).encode(
        x=alt.X("month:Q", title="Month"), y=alt.Y("balance:Q", title="Cumulative cash position (₹ lakh)"),
        tooltip=["month", alt.Tooltip("balance:Q", format=",.1f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    flows = df.melt(id_vars="month", value_vars=["cash_in", "cash_out"], var_name="flow", value_name="amount")
    bars = alt.Chart(flows).mark_bar(opacity=0.8).encode(
        x=alt.X("month:O", title="Month"), xOffset="flow:N", y=alt.Y("amount:Q", title="₹ lakh"),
        color=alt.Color("flow:N", scale=alt.Scale(domain=["cash_in", "cash_out"], range=["#2f7d6d", "#c2553a"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["month", "flow", alt.Tooltip("amount:Q", format=",.1f")])
    l, r = st.columns(2)
    l.markdown("**Cumulative cash position** (below zero = overdraft)")
    l.altair_chart((bal + zero).properties(height=320), width="stretch")
    r.markdown("**Cash in vs cash out each month**")
    r.altair_chart(bars.properties(height=320), width="stretch")

    st.markdown("#### What each clause costs the contractor")
    st.caption("Each row re-runs the forecast with only that clause made neutral (no retention, immediate payment, "
               "no BG, full price variation, and so on). Positive = the clause costs you money. "
               "Rows don't add up exactly because clauses interact.")
    show = imp.copy()
    l, r = st.columns([3, 2])
    l.dataframe(show, hide_index=True, width="stretch", column_config={
        "clause": "Clause",
        "pv_cost": st.column_config.NumberColumn("PV cost (₹ lakh)", format="%.2f"),
        "peak_funding_added": st.column_config.NumberColumn("Adds to peak funding (₹ lakh)", format="%.2f"),
        "od_interest_added": st.column_config.NumberColumn("Adds to overdraft interest (₹ lakh)", format="%.2f"),
    })
    r.altair_chart(alt.Chart(show).mark_bar().encode(
        x=alt.X("pv_cost:Q", title="PV cost (₹ lakh)"), y=alt.Y("clause:N", sort="-x", title=None),
        color=alt.condition("datum.pv_cost > 0", alt.value("#c2553a"), alt.value("#2f7d6d")),
        tooltip=["clause", alt.Tooltip("pv_cost:Q", format=",.2f")]).properties(height=260), width="stretch")
    st.info(f"**For the bid:** taken together, these terms cost **{lakh(combo['pv_cost'])} in present value, "
            f"{combo['pct_of_contract']:.1f}% of contract value**, compared with neutral terms (all clauses switched off "
            f"at once, so nothing is counted twice). Add it to the markup or negotiate the costliest clause first. "
            f"Plan an overdraft limit of at least {lakh(m['peak_funding'])}. "
            f"Maximum LD exposure if late: {lakh(m['max_ld_exposure'])}.")

    with st.expander("Monthly cash-flow table"):
        cols = ["month", "work_pct", "cumulative_work_pct", "work_value", "cost_paid", "gross_bill", "retention",
                "advance_recovery", "advance_interest", "ld", "net_bill", "receipt", "advance_in",
                "retention_release", "bg_margin", "bg_commission", "net_cash_flow", "od_interest", "balance"]
        st.dataframe(df[cols].round(2), hide_index=True, width="stretch")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df[cols].round(4).to_excel(xw, sheet_name="Monthly cash flow", index=False)
            pd.DataFrame([m]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="Summary")
            pd.DataFrame([asdict(terms)]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="Terms used")
            pd.DataFrame([asdict(p)]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="Project inputs")
            imp.to_excel(xw, sheet_name="Clause costs", index=False)
        st.download_button("Download forecast (Excel)", buf.getvalue(), file_name="clauseflow_forecast.xlsx")

# ------------------------------------------------------------------ 4. Sensitivity
with tab_sens:
    st.markdown("How peak funding moves with **payment delay** and **retention**, the two terms contractors most "
                "often try to negotiate. Other terms stay as verified.")
    metric = st.segmented_control("Show", ["peak_funding", "od_interest", "npv"], default="peak_funding",
                                  format_func={"peak_funding": "Peak funding", "od_interest": "Overdraft interest",
                                               "npv": "NPV"}.get)
    grid = sensitivity(p, terms, [0, 30, 60, 90, 120], [0.0, 2.5, 5.0, 7.5, 10.0])
    base = alt.Chart(grid).encode(x=alt.X("payment_days:O", title="Payment period (days)"),
                                  y=alt.Y("retention_pct:O", title="Retention (% of each bill)", sort="descending"))
    heat = base.mark_rect().encode(
        color=alt.Color(f"{metric}:Q", title="₹ lakh",
                        scale=alt.Scale(scheme="redyellowgreen", reverse=metric != "npv")),
        tooltip=["payment_days", "retention_pct", alt.Tooltip(f"{metric}:Q", format=",.1f")])
    text = base.mark_text(fontSize=13).encode(text=alt.Text(f"{metric}:Q", format=",.0f"))
    st.altair_chart((heat + text).properties(height=360), width="stretch")
    st.caption(f"Current contract: {terms.payment_days} days, {terms.retention_pct:g}% retention. Values in ₹ lakh.")

# ------------------------------------------------------------------ 5. Validation
with tab_val:
    st.markdown("Score the AI against a **manual extraction** of the same contract, done by a person reading it. "
                "Each clause is marked **correct**, **missed** or **incorrect** (wrong value or invented).")
    st.download_button("Download the manual-extraction template (CSV)", template().to_csv(index=False),
                       file_name="manual_extraction_template.csv")
    truth = st.file_uploader("Upload the completed manual extraction (CSV)", type=["csv"], key="truth")
    ex = st.session_state.get("extraction")
    if truth and not ex:
        st.warning("Load the AI extraction for the same contract in step ① first.")
    elif truth and ex:
        res, summ = score(ex, pd.read_csv(truth, dtype=str).fillna(""))
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{summ['accuracy'] * 100:.0f}%")
        b.metric("Correct", summ["correct"])
        c.metric("Missed", summ["missed"])
        d.metric("Incorrect", summ["incorrect"])
        st.dataframe(res, hide_index=True, width="stretch")
        st.download_button("Download scored results (CSV)", res.to_csv(index=False),
                           file_name=f"validation_{Path(ex.get('source', 'contract')).stem}.csv")

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown("""
#### How it works
1. **Extract.** The PDF's text is split by page and sent to Gemini with a fixed prompt asking for 8 clause types, each with a page number and a verbatim quote. Long documents are cut down to pages containing financial keywords.
2. **Citation check.** ClauseFlow searches the PDF text for each quote. Verbatim or near-match on the cited page = grounded. Anything else is flagged.
3. **Verify.** A person confirms or corrects every term. Only verified terms reach the forecast.
4. **Forecast.** Monthly model, month 0 = signing:
   - work done each month from the S-curve; contractor cost = cost % × work value × inflation index (1+i)^(t/12)
   - bills at the end of each billing cycle; gross bill = work value × (1 + price-variation coverage × (index − 1))
   - deductions: retention (to its cap), advance recovery (pro rata between the start and end progress %, balance from the final bill), interest on the outstanding advance, LD on the final bill
   - bills paid (payment days / 30) months later; a fractional lag is split between the two neighbouring months; retention released at completion and/or after the defect liability period
   - performance BG: cash margin blocked from signing to end of DLP, plus monthly commission
   - overdraft interest each month = rate/12 × previous month's negative balance; peak funding = deepest balance
   - NPV at the monthly equivalent of the annual cost of capital
5. **Clause cost.** Forecast re-run with one clause neutralised at a time; the NPV difference is that clause's PV cost.

#### Known limitations
- Monthly time steps: a 45-day payment period is modelled as half the bill after one month and half after two; exact day-by-day timing is ignored.
- LD is applied as a deduction on the final bill for a chosen delay; the schedule itself is not stretched.
- GST, TDS, secured advance on materials, advance bank guarantee charges and taxes are not modelled.
- Scanned PDFs need OCR first. Tables in PDFs sometimes extract out of order.
- Price variation is a single coverage % against a single inflation rate, not the contract's full index formula.
- The AI can misread clauses that look alike (e.g. damages for late site handover vs. damages for late completion). That is why step ② exists.
""")
