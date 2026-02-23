"""Core calculation engine for Polymarket take-profit strategy.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(Enum):
    HOLD = "Hold"
    PARTIAL_EXIT = "Partial Exit"
    FULL_EXIT = "Full Exit"
    STOP_LOSS = "Stop-Loss"


ACTION_COLORS = {
    Action.HOLD: "#2ecc71",
    Action.PARTIAL_EXIT: "#f39c12",
    Action.FULL_EXIT: "#e74c3c",
    Action.STOP_LOSS: "#c0392b",
}


@dataclass
class ContractInputs:
    p_market: float
    p_model: float
    days_to_resolution: int
    cost_basis: float | None = None
    position_size: float | None = None
    label: str = ""


@dataclass
class ContractResult:
    inputs: ContractInputs
    remaining_edge: float
    total_roi: float
    daily_roi: float
    action: Action
    reason: str
    sell_pct: float  # 0.0 = hold everything, 1.0 = sell everything
    unrealized_gain_pct: float | None = None
    net_edge_after_fees: float | None = None


def compute_daily_roi(
    p_market: float,
    p_model: float,
    days: int,
    compound: bool = False,
) -> float:
    """Compute expected daily ROI if held to resolution.

    Linear:   daily_roi = ((p_model - p_market) / p_market) / days
    Compound: daily_roi = (p_model / p_market)^(1/D) - 1
    """
    if p_market <= 0 or days <= 0:
        return 0.0
    total_roi = (p_model - p_market) / p_market
    if compound and p_model > 0 and p_market > 0:
        return (p_model / p_market) ** (1 / days) - 1
    return total_roi / days


def decide_action(
    p_market: float,
    p_model: float,
    days: int,
    hurdle: float = 0.002,
    cost_basis: float | None = None,
    compound: bool = False,
    fee_rate: float = 0.0,
) -> tuple[Action, str, float]:
    """Full decision cascade from KNOWLEDGEBASE.md.

    Returns (action, reason, sell_pct).
    """
    daily_roi = compute_daily_roi(p_market, p_model, days, compound)

    # 1. Edge gone (P_mod <= P_m)
    if p_model <= p_market:
        return Action.FULL_EXIT, "Edge gone — model ≤ market", 1.0

    # 2. Stop-loss: daily_ROI < -0.5%
    if daily_roi < -0.005:
        return Action.STOP_LOSS, f"Stop-loss triggered (daily ROI {daily_roi:.2%})", 1.0

    # 3. Unrealized gain >= 50% override
    if cost_basis is not None and cost_basis > 0:
        unrealized = (p_market - cost_basis) / cost_basis
        if unrealized >= 0.50:
            return (
                Action.PARTIAL_EXIT,
                f"Unrealized gain {unrealized:.0%} ≥ 50% — lock profits",
                0.40,  # sell 30-50%, use 40% midpoint
            )

    # 4. Near resolution (D <= 5) — lower effective hurdle (bias hold)
    effective_hurdle = hurdle
    if days <= 5:
        effective_hurdle = hurdle * 0.5

    # 5. Net edge after fees <= 0
    remaining_edge = p_model - p_market
    net_edge = remaining_edge - (fee_rate * p_market)
    if net_edge <= 0 and remaining_edge > 0:
        return Action.FULL_EXIT, "Net edge after fees ≤ 0", 1.0

    # 6. daily_ROI < hurdle → full exit
    if daily_roi < effective_hurdle:
        return (
            Action.FULL_EXIT,
            f"Daily ROI {daily_roi:.2%} < hurdle {effective_hurdle:.2%}",
            1.0,
        )

    # 7. daily_ROI < 1.5x hurdle → partial exit (50-70%, interpolated)
    if daily_roi < 1.5 * effective_hurdle:
        # Interpolate sell_pct: at hurdle → 70%, at 1.5×hurdle → 50%
        t = (daily_roi - effective_hurdle) / (0.5 * effective_hurdle) if effective_hurdle > 0 else 0
        sell_pct = 0.70 - 0.20 * t  # 70% at bottom, 50% at top
        sell_pct = max(0.50, min(0.70, sell_pct))
        return (
            Action.PARTIAL_EXIT,
            f"Daily ROI {daily_roi:.2%} in partial zone ({effective_hurdle:.2%}–{1.5 * effective_hurdle:.2%})",
            sell_pct,
        )

    # 8. Hold
    return Action.HOLD, f"Daily ROI {daily_roi:.2%} ≥ {1.5 * effective_hurdle:.2%} — hold", 0.0


def analyze_contract(
    inputs: ContractInputs,
    hurdle: float = 0.002,
    compound: bool = False,
    fee_rate: float = 0.0,
) -> ContractResult:
    """Orchestrator: compute all metrics and return a ContractResult."""
    p_m = inputs.p_market
    p_mod = inputs.p_model
    d = inputs.days_to_resolution

    remaining_edge = max(0.0, p_mod - p_m)
    total_roi = (p_mod - p_m) / p_m if p_m > 0 else 0.0
    daily_roi = compute_daily_roi(p_m, p_mod, d, compound)

    action, reason, sell_pct = decide_action(
        p_m, p_mod, d, hurdle, inputs.cost_basis, compound, fee_rate
    )

    unrealized = None
    if inputs.cost_basis is not None and inputs.cost_basis > 0:
        unrealized = (p_m - inputs.cost_basis) / inputs.cost_basis

    net_edge = remaining_edge - (fee_rate * p_m) if fee_rate > 0 else None

    return ContractResult(
        inputs=inputs,
        remaining_edge=remaining_edge,
        total_roi=total_roi,
        daily_roi=daily_roi,
        action=action,
        reason=reason,
        sell_pct=sell_pct,
        unrealized_gain_pct=unrealized,
        net_edge_after_fees=net_edge,
    )


def daily_roi_sensitivity(
    p_model: float,
    days: int,
    compound: bool = False,
    p_market_range: tuple[float, float] = (0.01, 0.99),
    steps: int = 200,
) -> list[dict]:
    """Generate data for sensitivity chart: daily_roi vs market price."""
    lo, hi = p_market_range
    step = (hi - lo) / steps
    rows = []
    for i in range(steps + 1):
        p_m = lo + i * step
        d_roi = compute_daily_roi(p_m, p_model, days, compound)
        rows.append({"p_market": round(p_m, 4), "daily_roi": d_roi})
    return rows


def _converge(initial: float, outcome: float, d_remaining: int, d_max: int, alpha: float) -> float:
    """Smoothly converge a probability toward the outcome as days remaining shrinks.

    At d_remaining == d_max → returns initial.
    At d_remaining → 0    → returns outcome.
    alpha controls speed: <1 = fast early convergence, >1 = slow then fast.
    """
    if d_max <= 0:
        return outcome
    t = d_remaining / d_max  # 1.0 at start, 0.0 at resolution
    blend = t ** alpha
    return outcome + (initial - outcome) * blend


def dynamic_p_model(
    p_model_init: float, outcome: float, d_remaining: int, d_max: int, alpha: float = 0.6,
) -> float:
    """Model probability that becomes more accurate as resolution approaches."""
    return max(0.01, min(0.99, _converge(p_model_init, outcome, d_remaining, d_max, alpha)))


def dynamic_p_market(
    p_market_init: float, outcome: float, d_remaining: int, d_max: int, alpha: float = 1.2,
) -> float:
    """Market probability that asymptotically converges to the outcome."""
    return max(0.01, min(0.99, _converge(p_market_init, outcome, d_remaining, d_max, alpha)))


def daily_roi_surface(
    p_model_init: float,
    p_market_init: float = 0.55,
    hurdle: float = 0.002,
    compound: bool = False,
    outcome: float = 1.0,
    d_max: int = 90,
    model_alpha: float = 0.6,
    market_alpha: float = 1.2,
    price_steps: int = 100,
    day_steps: int = 90,
) -> tuple[list[float], list[int], list[list[float]], list[list[int]], list[dict]]:
    """Generate surface data with dynamic probabilities over a countdown.

    Y axis is days remaining (counting down from d_max to 1).
    P_model converges to outcome faster (alpha=0.6) — model gets more accurate.
    P_market converges to outcome slower (alpha=1.2) — market catches up asymptotically.

    Returns (p_market_vals, days_remaining_vals, z_matrix, action_matrix, trajectory)
    where trajectory is a list of dicts for the expected market path line.
    """
    # X axis: range of possible market prices
    p_vals = [0.01 + i * 0.98 / price_steps for i in range(price_steps + 1)]

    # Y axis: days remaining, counting DOWN from d_max to 1
    step = max(1, d_max // day_steps)
    d_vals = list(range(d_max, 0, -step))
    if d_vals[-1] != 1:
        d_vals.append(1)

    action_map = {Action.HOLD: 0, Action.PARTIAL_EXIT: 1, Action.FULL_EXIT: 2, Action.STOP_LOSS: 3}

    z = []
    actions = []
    trajectory = []

    for d_rem in d_vals:
        # Dynamic model probability at this time step
        p_mod_d = dynamic_p_model(p_model_init, outcome, d_rem, d_max, model_alpha)
        # Expected market position at this time step (for trajectory line)
        p_mkt_d = dynamic_p_market(p_market_init, outcome, d_rem, d_max, market_alpha)

        z_row = []
        a_row = []
        for p_m in p_vals:
            roi = compute_daily_roi(p_m, p_mod_d, d_rem, compound)
            z_row.append(roi)
            act, _, _ = decide_action(p_m, p_mod_d, d_rem, hurdle, compound=compound)
            a_row.append(action_map[act])
        z.append(z_row)
        actions.append(a_row)

        # Trajectory: the expected daily ROI at the expected market price
        traj_roi = compute_daily_roi(p_mkt_d, p_mod_d, d_rem, compound)
        trajectory.append({
            "d_remaining": d_rem,
            "p_market": p_mkt_d,
            "p_model": p_mod_d,
            "daily_roi": traj_roi,
        })

    return p_vals, d_vals, z, actions, trajectory
