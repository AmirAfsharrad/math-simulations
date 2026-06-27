"""Linearization: tangent line (1D) and tangent plane (2D).

Authoritative Python behind the ``linearization`` browser demo. For a smooth f,
the best local linear model at a base point is

    1D:  f(x)    ~= f(x0)   + f'(x0) (x - x0)
    2D:  f(x, y) ~= f(x0,y0) + fx (x - x0) + fy (y - y0)

Derivatives are estimated by central differences, exactly as the demo does in
JavaScript. This file is the source of truth; keep the demo in sync with it.
"""
from __future__ import annotations

import numpy as np


def derivative(f, x0, h=1e-5):
    """Central-difference estimate of f'(x0)."""
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def tangent_line(f, x0, h=1e-5):
    """Tangent to f at x0 as (slope, intercept) for y = slope*x + intercept."""
    m = derivative(f, x0, h)
    return m, f(x0) - m * x0


def gradient(f, x0, y0, h=1e-5):
    """Central-difference gradient (f_x, f_y) of f at (x0, y0)."""
    fx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
    fy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)
    return fx, fy


def tangent_plane(f, x0, y0, h=1e-5):
    """Return P(x, y), the tangent plane to f at (x0, y0)."""
    f0 = f(x0, y0)
    fx, fy = gradient(f, x0, y0, h)

    def P(x, y):
        return f0 + fx * (x - x0) + fy * (y - y0)

    return P


# --- function libraries matching the demo dropdowns -------------------------
FUNCS_1D = {
    "sin": np.sin,
    "cubic": lambda x: x ** 3 / 3 - x,
    "exp": lambda x: np.exp(0.7 * x),
    "bump": lambda x: np.exp(-x ** 2),
}

FUNCS_2D = {
    "sincos": lambda x, y: np.sin(x) * np.cos(y),
    "saddle": lambda x, y: 0.5 * (x ** 2 - y ** 2),
    "bump": lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2),
    "xy": lambda x, y: x * y,
}
