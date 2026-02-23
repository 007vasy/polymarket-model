"""Polymarket Take-Profit Strategy — Streamlit App."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calc import (
    Action,
    ACTION_COLORS,
    ContractInputs,
    analyze_contract,
    compute_daily_roi,
    daily_roi_sensitivity,
    daily_roi_surface,
)
from portfolio import PortfolioConfig, analyze_portfolio, portfolio_summary

st.set_page_config(page_title="Polymarket Take-Profit", layout="wide")
st.title("Polymarket Take-Profit Strategy")

tab1, tab2 = st.tabs(["Single Contract", "Portfolio Dashboard"])

# ── Tab 1: Single Contract ──────────────────────────────────────────────────

with tab1:
    with st.sidebar:
        st.header("Contract Inputs")
        p_market = st.slider("Market Probability (P_market)", 0.01, 0.99, 0.55, 0.01)
        p_model = st.slider("Model Probability (P_model)", 0.01, 0.99, 0.70, 0.01)
        days = st.number_input("Days to Resolution", min_value=1, max_value=365, value=30)
        cost_basis = st.number_input(
            "Cost Basis (optional)", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
            help="Your average purchase price. Set to 0 to ignore.",
        )
        st.divider()
        st.header("Strategy Settings")
        hurdle = st.slider("Hurdle Rate (% / day)", 0.05, 1.0, 0.20, 0.05)
        hurdle_dec = hurdle / 100
        compound = st.toggle("Compound ROI formula", value=False)

    inputs = ContractInputs(
        p_market=p_market,
        p_model=p_model,
        days_to_resolution=days,
        cost_basis=cost_basis if cost_basis > 0 else None,
    )
    result = analyze_contract(inputs, hurdle_dec, compound)

    # Recommendation card
    color = ACTION_COLORS[result.action]
    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left:6px solid {color};
                    padding:16px; border-radius:6px; margin-bottom:16px;">
            <h2 style="margin:0; color:{color};">{result.action.value}</h2>
            <p style="margin:4px 0 0;">{result.reason}</p>
            {"<p>Sell " + f"{result.sell_pct:.0%}" + " of position</p>" if result.sell_pct > 0 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Remaining Edge", f"{result.remaining_edge:.2%}")
    m2.metric("Total ROI", f"{result.total_roi:.2%}")
    m3.metric("Daily ROI", f"{result.daily_roi:.2%}")
    if result.unrealized_gain_pct is not None:
        m4.metric("Unrealized Gain", f"{result.unrealized_gain_pct:.1%}")
    else:
        m4.metric("Unrealized Gain", "N/A")

    # 3D Option Surface
    st.subheader("Daily ROI Surface: Market Price × Days Remaining")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        max_days = st.slider("Max days", 10, 180, max(days * 2, 60), 10, key="surf_max_d")
    with sc2:
        outcome = st.radio("Assumed outcome", [1.0, 0.0], format_func=lambda v: "Yes ($1)" if v == 1.0 else "No ($0)", key="outcome", horizontal=True)
    with sc3:
        model_alpha = st.slider("Model convergence speed", 0.2, 2.0, 0.6, 0.1, key="m_alpha",
                                help="Lower = model gets accurate faster")
        market_alpha = st.slider("Market convergence speed", 0.2, 3.0, 1.2, 0.1, key="mkt_alpha",
                                 help="Higher = market lags behind model")

    p_vals, d_vals, z_matrix, action_matrix, trajectory = daily_roi_surface(
        p_model_init=p_model,
        p_market_init=p_market,
        hurdle=hurdle_dec,
        compound=compound,
        outcome=outcome,
        d_max=max_days,
        model_alpha=model_alpha,
        market_alpha=market_alpha,
        price_steps=80,
        day_steps=min(max_days, 80),
    )

    import numpy as np
    z_arr = np.array(z_matrix)
    act_arr = np.array(action_matrix)

    # Clamp z for display (cap extreme values near p_market→0)
    z_cap = float(np.percentile(z_arr, 98)) * 1.5
    z_display = np.clip(z_arr, None, z_cap)

    fig = go.Figure()

    # Main surface colored by action zone
    fig.add_trace(go.Surface(
        x=p_vals,
        y=d_vals,
        z=z_display,
        surfacecolor=act_arr,
        colorscale=[
            [0.0, "#2ecc71"],   # hold
            [0.33, "#f39c12"],  # partial
            [0.67, "#e74c3c"],  # full exit
            [1.0, "#c0392b"],   # stop-loss
        ],
        cmin=0, cmax=3,
        colorbar=dict(
            title="Action",
            tickvals=[0, 1, 2, 3],
            ticktext=["Hold", "Partial", "Full Exit", "Stop-Loss"],
        ),
        opacity=0.85,
        name="Daily ROI",
        hovertemplate="P_market: %{x:.2f}<br>Days remaining: %{y}<br>Daily ROI: %{z:.2%}<extra></extra>",
    ))

    # Hurdle plane
    hurdle_z = [[hurdle_dec] * len(p_vals)] * len(d_vals)
    fig.add_trace(go.Surface(
        x=p_vals, y=d_vals, z=hurdle_z,
        colorscale=[[0, "rgba(231,76,60,0.3)"], [1, "rgba(231,76,60,0.3)"]],
        showscale=False, opacity=0.25, name="Hurdle Plane",
        hovertemplate="Hurdle: %{z:.2%}<extra></extra>",
    ))

    # Expected trajectory line (model predicts market path)
    traj_x = [t["p_market"] for t in trajectory]
    traj_y = [t["d_remaining"] for t in trajectory]
    traj_z = [min(t["daily_roi"], z_cap) for t in trajectory]
    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z,
        mode="lines+markers",
        line=dict(color="#ffffff", width=5),
        marker=dict(size=2, color="#ffffff"),
        name="Expected Path",
        hovertemplate="Day %{y} remaining<br>P_mkt: %{x:.2f}<br>Daily ROI: %{z:.2%}<extra></extra>",
    ))

    # Current position marker
    current_roi = compute_daily_roi(p_market, p_model, days, compound)
    fig.add_trace(go.Scatter3d(
        x=[p_market], y=[days], z=[min(current_roi, z_cap)],
        mode="markers",
        marker=dict(size=8, color=color, symbol="diamond"),
        name="Current Position",
        hovertemplate=f"P_m={p_market:.2f}, D={days}<br>Daily ROI: {current_roi:.2%}<extra></extra>",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Market Price",
            yaxis_title="Days Remaining",
            yaxis=dict(autorange="reversed"),  # countdown: high at back, low at front
            zaxis_title="Daily ROI",
            zaxis_tickformat=".1%",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.0)),
        ),
        height=1600,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Portfolio Dashboard ───────────────────────────────────────────────

with tab2:
    with st.sidebar:
        st.header("Portfolio Config")
        total_book = st.number_input("Total Book Value ($)", value=10000.0, step=500.0)
        cash_buffer = st.slider("Cash Buffer %", 0, 50, 25, 5)
        max_pos = st.number_input("Max Positions", value=10, min_value=1, max_value=50)
        max_per_trade = st.slider("Max Per-Trade %", 1, 20, 5, 1)
        portfolio_hurdle = st.slider(
            "Portfolio Hurdle (% / day)", 0.05, 1.0, 0.20, 0.05, key="port_hurdle",
        )
        portfolio_hurdle_dec = portfolio_hurdle / 100
        port_compound = st.toggle("Compound (portfolio)", value=False, key="port_compound")

    config = PortfolioConfig(
        total_book_value=total_book,
        cash_buffer_pct=cash_buffer / 100,
        max_positions=max_pos,
        max_per_trade_pct=max_per_trade / 100,
        hurdle_rate=portfolio_hurdle_dec,
    )

    st.subheader("Enter Positions")
    default_data = pd.DataFrame(
        {
            "Label": ["Contract A", "Contract A (moved)", "Contract B", "Contract C"],
            "P_market": [0.55, 0.68, 0.55, 0.59],
            "P_model": [0.70, 0.70, 0.70, 0.60],
            "Days": [30, 25, 5, 10],
            "Cost Basis": [0.0, 0.0, 0.0, 0.0],
            "Position ($)": [500.0, 500.0, 500.0, 500.0],
        }
    )

    edited_df = st.data_editor(
        default_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "P_market": st.column_config.NumberColumn(min_value=0.01, max_value=0.99, step=0.01),
            "P_model": st.column_config.NumberColumn(min_value=0.01, max_value=0.99, step=0.01),
            "Days": st.column_config.NumberColumn(min_value=1, max_value=365, step=1),
            "Cost Basis": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
            "Position ($)": st.column_config.NumberColumn(min_value=0.0, step=50.0),
        },
    )

    if st.button("Analyze Portfolio", type="primary"):
        contracts = []
        for _, row in edited_df.iterrows():
            cb = row["Cost Basis"] if row["Cost Basis"] > 0 else None
            contracts.append(
                ContractInputs(
                    p_market=row["P_market"],
                    p_model=row["P_model"],
                    days_to_resolution=int(row["Days"]),
                    cost_basis=cb,
                    position_size=row["Position ($)"],
                    label=row["Label"],
                )
            )

        results_df = analyze_portfolio(contracts, config, port_compound)
        summary = portfolio_summary(results_df, config)

        # Summary metrics
        st.subheader("Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Invested", f"${summary['total_invested']:,.0f}")
        s2.metric("Cash Available", f"${summary['cash']:,.0f}")
        s3.metric("Wtd Avg Daily ROI", f"{summary['weighted_avg_daily_roi']:.2%}")
        s4.metric("Capital Freed", f"${summary['capital_freed']:,.0f}")

        # Results table
        st.subheader("Results (sorted by Daily ROI — exit worst first)")
        display_df = results_df.drop(columns=["_action_enum", "_color"])

        def color_action(val: str) -> str:
            colors = {
                "Hold": "background-color: #2ecc7133",
                "Partial Exit": "background-color: #f39c1233",
                "Full Exit": "background-color: #e74c3c33",
                "Stop-Loss": "background-color: #c0392b33",
            }
            return colors.get(val, "")

        styled = display_df.style.map(color_action, subset=["Action"]).format(
            {
                "P_market": "{:.2%}",
                "P_model": "{:.2%}",
                "Edge": "{:.2%}",
                "Total ROI": "{:.2%}",
                "Daily ROI": "{:.2%}",
                "Sell %": "{:.0%}",
                "Cost Basis": lambda v: f"{v:.2f}" if pd.notna(v) and v else "—",
                "Position ($)": "${:,.0f}",
            }
        )
        st.dataframe(styled, use_container_width=True)

        # Bar chart
        st.subheader("Daily ROI by Position")
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            y=results_df["Label"],
            x=results_df["Daily ROI"],
            orientation="h",
            marker_color=results_df["_color"],
            text=results_df["Daily ROI"].apply(lambda v: f"{v:.2%}"),
            textposition="outside",
        ))
        bar_fig.add_vline(x=config.hurdle_rate, line_dash="dash", line_color="#e74c3c",
                          annotation_text=f"Hurdle ({portfolio_hurdle:.2f}%)")
        bar_fig.update_layout(
            xaxis_title="Daily ROI",
            xaxis_tickformat=".2%",
            height=max(300, len(results_df) * 60),
            margin=dict(l=20, r=20, t=30, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)
