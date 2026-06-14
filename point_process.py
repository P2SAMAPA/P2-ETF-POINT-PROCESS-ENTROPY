import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def discretise(series, n_bins):
    """Equal‑frequency discretisation (for macro)."""
    if len(series) < n_bins:
        return np.zeros_like(series, dtype=int)
    quantiles = np.linspace(0, 100, n_bins+1)[1:-1]
    bins = np.percentile(series, quantiles)
    return np.digitize(series, bins)

def binary_sequence(returns):
    """Convert returns to binary: 1 = up, 0 = down (or flat)."""
    b = np.zeros(len(returns))
    b[returns > 0] = 1
    b[returns < 0] = 0
    return b

def run_entropy(seq):
    """Compute entropy of run lengths (consecutive same symbols)."""
    if len(seq) == 0:
        return 0.0
    runs = []
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    if len(runs) == 0:
        return 0.0
    # Entropy of run lengths
    counts = np.bincount(runs)
    probs = counts[counts > 0] / len(runs)
    return entropy(probs, base=2)

def betting_correlation(seq, lag=1):
    """
    Compute betting correlation: correlation between past and future outcomes.
    For binary sequence, this is the phi coefficient.
    """
    if len(seq) < lag + 1:
        return 0.0
    past = seq[:-lag]
    future = seq[lag:]
    # Contingency table
    n11 = np.sum((past == 1) & (future == 1))
    n10 = np.sum((past == 1) & (future == 0))
    n01 = np.sum((past == 0) & (future == 1))
    n00 = np.sum((past == 0) & (future == 0))
    n = n11 + n10 + n01 + n00
    if n == 0:
        return 0.0
    phi = (n11 * n00 - n10 * n01) / np.sqrt((n11+n10)*(n11+n01)*(n00+n10)*(n00+n01) + 1e-8)
    return phi

def point_process_score(returns, macro_df, macro_bins=3):
    """
    Compute score = betting correlation weighted by macro‑conditioned run entropy.
    For each macro variable, compute separate betting correlation for each macro bin,
    then combine with macro importance weights (ridge regression).
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Convert to binary
    binary = binary_sequence(returns)
    # For each macro, compute betting correlation per bin
    per_macro_scores = []
    for col in macro_df.columns:
        macro_series = macro_df[col].values
        macro_disc = discretise(macro_series, macro_bins)
        # For each macro bin, compute betting correlation
        bc_by_bin = []
        for b in range(macro_bins):
            idx = (macro_disc == b)
            if np.sum(idx) < 10:
                bc = 0.0
            else:
                bc = betting_correlation(binary[idx], lag=1)
            bc_by_bin.append(bc)
        # Score = difference between highest and lowest bin
        score = max(bc_by_bin) - min(bc_by_bin)
        per_macro_scores.append(score)
    # Estimate macro importance via ridge regression (predict next‑day return)
    X = macro_df.iloc[:-1].values
    y = returns[1:]
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(y) < 10:
        return float(np.mean(per_macro_scores))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    weights = np.abs(ridge.coef_)
    if weights.sum() < 1e-8:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weights.sum()
    combined_score = np.sum(weights * np.array(per_macro_scores))
    # Clip to reasonable range
    return float(max(-1.0, min(1.0, combined_score)))
