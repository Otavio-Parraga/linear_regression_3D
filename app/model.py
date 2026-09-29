"""Univariate linear regression: hypothesis, cost, gradient and cost-surface grid.

All functions are pure and NumPy-based.
    h(x) = t0 + t1 * x
    J(t0, t1) = (1 / (2m)) * sum((h(x) - y)^2)
"""

from __future__ import annotations

import numpy as np

# Fixed dataset: hours studied (x) vs exam score (y).
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
Y = np.array([46.1, 49.8, 56.3, 58.9, 66.2, 69.1, 74.8, 79.5, 83.2, 91.0], dtype=float)

GRID_N = 101
# Half-widths of the grid window (t0, t1) around theta_star, per scale.
HALF_WIDTH = {
    "original": (30.0, 6.0),
    "normalized": (30.0, 30.0),
}
DECIMALS = 6


def predict(x, t0: float, t1: float) -> np.ndarray:
    return t0 + t1 * np.asarray(x, dtype=float)


def cost(x, y, t0: float, t1: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = predict(x, t0, t1) - y
    return float(np.sum(r * r) / (2 * x.size))


def gradient(x, y, t0: float, t1: float) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = predict(x, t0, t1) - y
    return float(np.mean(r)), float(np.mean(r * x))


def normal_equation(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    theta = np.linalg.solve(A.T @ A, A.T @ y)
    return float(theta[0]), float(theta[1])


def cost_grid(x, y, t0_vals, t1_vals) -> np.ndarray:
    """J[i, j] = cost(t0 = t0_vals[j], t1 = t1_vals[i]) (Plotly surface convention)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t0 = np.asarray(t0_vals, dtype=float)
    t1 = np.asarray(t1_vals, dtype=float)
    # shape (len(t1), len(t0), m)
    h = t0[None, :, None] + t1[:, None, None] * x[None, None, :]
    r = h - y[None, None, :]
    return np.sum(r * r, axis=-1) / (2 * x.size)


def normalize(x) -> tuple[np.ndarray, float, float]:
    """z = (x - mean) / std using population std (ddof=0)."""
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    return (x - mean) / std, mean, std


def _round(a):
    return np.round(np.asarray(a, dtype=float), DECIMALS).tolist()


def _scale_payload(feature: np.ndarray, y: np.ndarray, half_widths: tuple[float, float]) -> dict:
    t0s, t1s = normal_equation(feature, y)
    w0, w1 = half_widths
    r0 = (t0s - w0, t0s + w0)
    r1 = (t1s - w1, t1s + w1)
    g0 = np.linspace(*r0, GRID_N)
    g1 = np.linspace(*r1, GRID_N)
    J = cost_grid(feature, y, g0, g1)
    return {
        "feature": _round(feature),
        "theta_star": _round([t0s, t1s]),
        "J_star": round(cost(feature, y, t0s, t1s), DECIMALS),
        "range": {"t0": _round(r0), "t1": _round(r1)},
        "grid": {"t0": _round(g0), "t1": _round(g1), "J": _round(J)},
    }


def build_model_payload(x=X, y=Y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z, mean, std = normalize(x)
    return {
        "data": {"x": _round(x), "y": _round(y)},
        "stats": {"mean": round(mean, DECIMALS), "std": round(std, DECIMALS)},
        "scales": {
            "original": _scale_payload(x, y, HALF_WIDTH["original"]),
            "normalized": _scale_payload(z, y, HALF_WIDTH["normalized"]),
        },
    }
