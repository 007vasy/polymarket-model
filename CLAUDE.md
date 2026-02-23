# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymarket take-profit strategy tool. A Streamlit app that recommends hold/partial-exit/full-exit for prediction market positions to maximize daily book value returns. See `KNOWLEDGEBASE.md` for the underlying strategy.

## Commands

```bash
# Run the app
uv run streamlit run app.py

# Run a quick calculation check
uv run python -c "from calc import *; r = analyze_contract(ContractInputs(0.55, 0.70, 30)); print(r)"
```

## Architecture

- `calc.py` — Pure calculation engine (no UI deps). `Action` enum, `ContractInputs`/`ContractResult` dataclasses, `compute_daily_roi`, `decide_action`, `analyze_contract`, `daily_roi_sensitivity`.
- `portfolio.py` — Portfolio aggregation. `PortfolioConfig`, `analyze_portfolio` (returns sorted DataFrame), `portfolio_summary`, `capital_freed`.
- `app.py` — Streamlit UI. Tab 1: single contract analysis + Plotly sensitivity chart. Tab 2: portfolio dashboard with data editor + bar chart.
