"""Portfolio-level aggregation for the take-profit strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from calc import Action, ContractInputs, ContractResult, analyze_contract, ACTION_COLORS


@dataclass
class PortfolioConfig:
    total_book_value: float = 10_000.0
    cash_buffer_pct: float = 0.25
    max_positions: int = 10
    max_per_trade_pct: float = 0.05
    hurdle_rate: float = 0.002


def analyze_portfolio(
    contracts: list[ContractInputs],
    config: PortfolioConfig,
    compound: bool = False,
    fee_rate: float = 0.0,
) -> pd.DataFrame:
    """Analyze all contracts and return a DataFrame sorted by daily_roi ascending (exit worst first)."""
    results: list[ContractResult] = []
    for c in contracts:
        r = analyze_contract(c, config.hurdle_rate, compound, fee_rate)
        results.append(r)

    rows = []
    for r in results:
        rows.append(
            {
                "Label": r.inputs.label or f"P_mod={r.inputs.p_model:.0%}",
                "P_market": r.inputs.p_market,
                "P_model": r.inputs.p_model,
                "Days": r.inputs.days_to_resolution,
                "Cost Basis": r.inputs.cost_basis,
                "Position ($)": r.inputs.position_size,
                "Edge": r.remaining_edge,
                "Total ROI": r.total_roi,
                "Daily ROI": r.daily_roi,
                "Action": r.action.value,
                "Sell %": r.sell_pct,
                "Reason": r.reason,
                "_action_enum": r.action,
                "_color": ACTION_COLORS[r.action],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Daily ROI", ascending=True).reset_index(drop=True)
    return df


def portfolio_summary(df: pd.DataFrame, config: PortfolioConfig) -> dict:
    """Compute summary statistics from portfolio analysis results."""
    total_invested = df["Position ($)"].sum() if "Position ($)" in df.columns else 0.0
    total_invested = total_invested if pd.notna(total_invested) else 0.0
    cash = config.total_book_value - total_invested

    weights = df["Position ($)"].infer_objects(copy=False).fillna(0)
    total_w = weights.sum()
    if total_w > 0:
        weighted_daily_roi = (df["Daily ROI"] * weights).sum() / total_w
    else:
        weighted_daily_roi = df["Daily ROI"].mean() if not df.empty else 0.0

    freed = capital_freed(df)

    return {
        "total_invested": total_invested,
        "cash": cash,
        "weighted_avg_daily_roi": weighted_daily_roi,
        "capital_freed": freed,
        "num_positions": len(df),
        "num_exits": len(df[df["Action"].isin(["Full Exit", "Stop-Loss"])]),
        "num_partial": len(df[df["Action"] == "Partial Exit"]),
        "num_holds": len(df[df["Action"] == "Hold"]),
    }


def capital_freed(df: pd.DataFrame) -> float:
    """Calculate USDC freed for redeployment based on recommended actions."""
    freed = 0.0
    for _, row in df.iterrows():
        pos = row.get("Position ($)")
        if pos is None or pd.isna(pos):
            continue
        freed += pos * row["Sell %"]
    return freed
