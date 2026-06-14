# Point‑Process Entropy (Betting Correlation) for ETFs

Converts ETF returns into binary up/down events. Computes betting correlation (predictability of direction) separately for macro regimes (low/medium/high). The per‑ETF score is the weighted average of regime‑contrast across all macro variables – a measure of directional predictability.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Binary event sequence: 1 (up), 0 (down)
- Betting correlation (phi coefficient) for lag‑1 dependence
- Macro stratification into low/medium/high bins
- Ridge regression weights for macro importance
- Score = weighted average of betting correlation differences across macro regimes
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-point-process-entropy-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF’s direction is highly predictable given current macro regime.
- Low score → direction is unpredictable (random walk).

## Requirements

See `requirements.txt`.
