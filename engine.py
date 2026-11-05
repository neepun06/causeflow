"""ClauseFlow cash-flow engine.

Turns verified contract terms plus a project schedule into a monthly cash-flow
forecast for the contractor. Pure Python + pandas, no AI involved, so every
number here can be reproduced by hand or in a plain spreadsheet.

Conventions (all documented in the user manual):
- Month 0 = contract signing. Work is done in months 1..N.
- Money is in Rs lakh. Rates are % per annum unless named otherwise.
- Bills are raised at the end of each billing cycle and at completion (month N).
- A bill is paid payment_days / 30 months after it is raised. A fractional lag is split
  between the two neighbouring months (45 days = half after 1 month, half after 2), so the
  forecast has no jump when the payment period crosses a month boundary.
- Overdraft interest is charged monthly on the previous month's negative balance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace

import pandas as pd


@dataclass
class ClauseTerms:
    """The financial terms ClauseFlow extracts from a contract."""
    advance_pct: float = 10.0                 # mobilisation advance, % of contract value
    advance_interest_pct: float = 10.0        # simple interest on outstanding advance, % p.a.
    advance_recovery_start_pct: float = 10.0  # recovery starts at this cumulative progress %
    advance_recovery_end_pct: float = 80.0    # fully recovered by this cumulative progress %
    retention_pct: float = 5.0                # % deducted from each gross bill
    retention_cap_pct: float = 5.0            # cap on total retention, % of contract value (0 = no cap)
    retention_release_completion_pct: float = 0.0  # share of retention released at completion
    billing_cycle_months: int = 1
    payment_days: int = 30
    ld_rate_pct_month: float = 1.5            # liquidated damages, % of contract value per month of delay
    ld_cap_pct: float = 10.0                  # LD cap, % of contract value
    pv_coverage_pct: float = 0.0              # share of cost escalation recovered via price variation (0 = no clause)
    pbg_pct: float = 5.0                      # performance bank guarantee, % of contract value
    dlp_months: int = 12                      # defect liability period


@dataclass
class ProjectInputs:
    contract_value: float = 1000.0            # Rs lakh
    duration_months: int = 12
    schedule_pct: list[float] = field(default_factory=list)  # % of work done each month, sums to 100
    cost_ratio_pct: float = 88.0              # contractor's cost as % of value of work done
    cost_lag_months: int = 0                  # supplier credit: costs paid this many months after incurred
    od_rate_pct: float = 11.0                 # overdraft interest, % p.a.
    discount_rate_pct: float = 12.0           # contractor's cost of capital for NPV, % p.a.
    inflation_pct: float = 6.0                # input cost escalation, % p.a.
    delay_months: float = 0.0                 # completion-delay scenario for LD
    bg_commission_pct: float = 1.5            # bank charge on guarantees, % p.a.
    bg_margin_pct: float = 25.0               # cash margin the bank blocks against the BG, % of BG

    def schedule(self) -> list[float]:
        return self.schedule_pct or s_curve(self.duration_months)


# Neutral value for each term: what the term would be if the clause did not hurt
# (or help) the contractor. Used to price each clause in isolation.
NEUTRAL = {
    "advance": dict(advance_pct=0.0, advance_interest_pct=0.0),
    "retention": dict(retention_pct=0.0),
    "billing_cycle": dict(billing_cycle_months=1),
    "payment_period": dict(payment_days=0),
    "liquidated_damages": dict(ld_rate_pct_month=0.0),
    "price_variation": dict(pv_coverage_pct=100.0),
    "bank_guarantee": dict(pbg_pct=0.0),
    "defect_liability": dict(dlp_months=0),
}

CLAUSE_LABELS = {
    "advance": "Mobilisation advance",
    "retention": "Retention",
    "billing_cycle": "Billing cycle",
    "payment_period": "Payment period",
    "liquidated_damages": "Liquidated damages",
    "price_variation": "Price variation",
    "bank_guarantee": "Performance bank guarantee",
    "defect_liability": "Defect liability period",
}


def s_curve(n: int) -> list[float]:
    """Monthly % of work for a standard S-curve (cumulative = 3x^2 - 2x^3), rounded to 2 dp."""
    cum = [100 * (3 * (t / n) ** 2 - 2 * (t / n) ** 3) for t in range(n + 1)]
    monthly = [round(cum[t] - cum[t - 1], 2) for t in range(1, n + 1)]
    monthly[-1] = round(100 - sum(monthly[:-1]), 2)
    return monthly


def linear_curve(n: int) -> list[float]:
    monthly = [round(100 / n, 2)] * n
    monthly[-1] = round(100 - sum(monthly[:-1]), 2)
    return monthly


def _recovered_fraction(progress: float, start: float, end: float) -> float:
    """Share of the advance that should have been recovered at a cumulative progress %."""
    if progress >= end:
        return 1.0
    if progress <= start:
        return 0.0
    return (progress - start) / (end - start)


def forecast(p: ProjectInputs, c: ClauseTerms) -> tuple[pd.DataFrame, dict]:
    """Monthly cash-flow table and headline metrics."""
    n = p.duration_months
    sched = p.schedule()
    if len(sched) != n:
        raise ValueError(f"Schedule has {len(sched)} months but duration is {n}.")
    if abs(sum(sched) - 100) > 0.05:
        raise ValueError(f"Schedule sums to {sum(sched):.2f}%, not 100%.")

    V = p.contract_value
    lag_f = max(c.payment_days, 0) / 30
    lag_lo, frac = int(math.floor(lag_f)), lag_f - math.floor(lag_f)
    lag = lag_lo + (1 if frac > 1e-9 else 0)  # last month a payment can land in
    cycle = max(int(c.billing_cycle_months), 1)
    dlp = max(int(c.dlp_months), 0)
    horizon = n + dlp + lag + p.cost_lag_months + 1
    months = range(0, horizon + 1)

    z = lambda: {t: 0.0 for t in months}  # noqa: E731

    def paid_later(arr: dict, month: int, amount: float) -> None:
        arr[month + lag_lo] += amount * (1 - frac)
        if frac > 1e-9:
            arr[month + lag_lo + 1] += amount * frac

    work = z(); cost_incurred = z(); cost_paid = z(); gross = z(); retention = z()
    adv_rec = z(); adv_int = z(); ld = z(); net_bill = z(); receipt = z()
    advance_in = z(); ret_release = z(); bg_margin = z(); bg_comm = z()

    # Work done and cost
    for t in range(1, n + 1):
        work[t] = V * sched[t - 1] / 100
        esc = (1 + p.inflation_pct / 100) ** (t / 12)
        cost_incurred[t] = work[t] * p.cost_ratio_pct / 100 * esc
        cost_paid[t + p.cost_lag_months] += cost_incurred[t]

    # Advance received at signing
    A = V * c.advance_pct / 100
    advance_in[0] = A

    # Bills
    bill_months = sorted(set(list(range(cycle, n + 1, cycle)) + [n]))
    prev_b, cum_pct, recovered = 0, 0.0, 0.0
    ret_total, ret_cap = 0.0, (V * c.retention_cap_pct / 100 if c.retention_cap_pct > 0 else float("inf"))
    outstanding = A
    for b in bill_months:
        covered = range(prev_b + 1, b + 1)
        g = 0.0
        interest = 0.0
        for m in covered:
            esc = (1 + p.inflation_pct / 100) ** (m / 12)
            g += work[m] * (1 + c.pv_coverage_pct / 100 * (esc - 1))
            interest += outstanding * c.advance_interest_pct / 100 / 12
            cum_pct += sched[m - 1]
        target = A * _recovered_fraction(cum_pct, c.advance_recovery_start_pct, c.advance_recovery_end_pct)
        if b == n:
            target = A  # anything left is recovered from the final bill
        rec = target - recovered
        recovered, outstanding = target, A - target
        r = min(g * c.retention_pct / 100, ret_cap - ret_total)
        ret_total += r
        ld_amt = 0.0
        if b == n:
            ld_amt = V * min(c.ld_rate_pct_month * p.delay_months, c.ld_cap_pct) / 100
        gross[b], retention[b], adv_rec[b], adv_int[b], ld[b] = g, r, rec, interest, ld_amt
        net_bill[b] = g - r - rec - interest - ld_amt
        paid_later(receipt, b, net_bill[b])
        prev_b = b

    # Retention release: part with the final payment, rest after the DLP
    share = c.retention_release_completion_pct / 100
    paid_later(ret_release, n, ret_total * share)
    paid_later(ret_release, n + dlp, ret_total * (1 - share))

    # Performance bank guarantee: cash margin blocked from signing to end of DLP, plus commission
    G = V * c.pbg_pct / 100
    bg_margin[0] -= G * p.bg_margin_pct / 100
    bg_margin[n + dlp] += G * p.bg_margin_pct / 100
    for t in range(1, n + dlp + 1):
        bg_comm[t] = G * p.bg_commission_pct / 100 / 12

    rows, bal = [], 0.0
    r_m = (1 + p.discount_rate_pct / 100) ** (1 / 12) - 1
    for t in months:
        inflow = receipt[t] + advance_in[t] + ret_release[t]
        outflow = cost_paid[t] + bg_comm[t]
        ncf = inflow - outflow + bg_margin[t]
        od_int = max(0.0, -bal) * p.od_rate_pct / 100 / 12
        bal = bal + ncf - od_int
        rows.append(dict(
            month=t, work_pct=sched[t - 1] if 1 <= t <= n else 0.0, work_value=work[t],
            cost_paid=cost_paid[t], gross_bill=gross[t], retention=retention[t],
            advance_recovery=adv_rec[t], advance_interest=adv_int[t], ld=ld[t], net_bill=net_bill[t],
            receipt=receipt[t], advance_in=advance_in[t], retention_release=ret_release[t],
            bg_margin=bg_margin[t], bg_commission=bg_comm[t], net_cash_flow=ncf,
            od_interest=od_int, balance=bal, discount_factor=1 / (1 + r_m) ** t,
        ))
    df = pd.DataFrame(rows)
    df["cumulative_work_pct"] = df["work_pct"].cumsum()
    df["cash_in"] = df["receipt"] + df["advance_in"] + df["retention_release"] + df["bg_margin"].clip(lower=0)
    df["cash_out"] = df["cost_paid"] + df["bg_commission"] - df["bg_margin"].clip(upper=0)

    peak_idx = df["balance"].idxmin()
    metrics = dict(
        peak_funding=max(0.0, -df["balance"].min()),
        peak_month=int(df.loc[peak_idx, "month"]) if df["balance"].min() < 0 else None,
        od_interest=df["od_interest"].sum(),
        npv=(df["net_cash_flow"] * df["discount_factor"]).sum(),
        profit=df["net_cash_flow"].sum(),
        profit_after_interest=df["net_cash_flow"].sum() - df["od_interest"].sum(),
        months_negative=int((df["balance"] < 0).sum()),
        max_ld_exposure=V * c.ld_cap_pct / 100 if c.ld_rate_pct_month > 0 else 0.0,
        total_retention=ret_total,
        payment_lag_months=lag_f,
    )
    return df, metrics


def clause_impacts(p: ProjectInputs, c: ClauseTerms) -> pd.DataFrame:
    """Price each clause by re-running the forecast with only that clause neutralised.

    Cost > 0 means the clause hurts the contractor's cash position relative to the neutral term.
    Costs do not add up exactly to the combined effect because clauses interact.
    """
    _, base = forecast(p, c)
    out = []
    for key, neutral in NEUTRAL.items():
        _, alt = forecast(p, replace(c, **neutral))
        out.append(dict(
            clause=CLAUSE_LABELS[key],
            pv_cost=alt["npv"] - base["npv"],
            peak_funding_added=base["peak_funding"] - alt["peak_funding"],
            od_interest_added=base["od_interest"] - alt["od_interest"],
        ))
    return pd.DataFrame(out)


def combined_cost(p: ProjectInputs, c: ClauseTerms) -> dict:
    """All clauses neutralised at once vs. the actual terms. Avoids double counting between clauses
    and between NPV and overdraft interest: this is the figure to carry into the bid markup."""
    neutral = {k: v for d in NEUTRAL.values() for k, v in d.items()}
    _, base = forecast(p, c)
    _, alt = forecast(p, replace(c, **neutral))
    return dict(pv_cost=alt["npv"] - base["npv"], pct_of_contract=(alt["npv"] - base["npv"]) / p.contract_value * 100,
                peak_funding_added=base["peak_funding"] - alt["peak_funding"])


def sensitivity(p: ProjectInputs, c: ClauseTerms, payment_days: list[int], retention_pcts: list[float]) -> pd.DataFrame:
    rows = []
    for d in payment_days:
        for r in retention_pcts:
            _, m = forecast(p, replace(c, payment_days=d, retention_pct=r))
            rows.append(dict(payment_days=d, retention_pct=r, peak_funding=m["peak_funding"],
                             od_interest=m["od_interest"], npv=m["npv"]))
    return pd.DataFrame(rows)


def terms_to_dict(c: ClauseTerms) -> dict:
    return asdict(c)
