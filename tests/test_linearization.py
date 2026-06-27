import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import linearization as lin  # noqa: E402


def test_derivative_of_sin():
    assert abs(lin.derivative(np.sin, 0.0) - 1.0) < 1e-6
    assert abs(lin.derivative(np.sin, np.pi / 2)) < 1e-6


def test_tangent_line_touches_curve():
    f = np.sin
    m, b = lin.tangent_line(f, 1.0)
    assert abs((m * 1.0 + b) - f(1.0)) < 1e-9


def test_gradient_matches_known():
    f = lambda x, y: x * x + 3 * y
    fx, fy = lin.gradient(f, 1.0, 2.0)
    assert abs(fx - 2.0) < 1e-5   # d/dx (x^2) = 2x = 2 at x=1
    assert abs(fy - 3.0) < 1e-5


def test_tangent_plane_matches_value_and_gradient():
    f = lambda x, y: np.sin(x) * np.cos(y)
    P = lin.tangent_plane(f, 0.6, -0.4)
    assert abs(P(0.6, -0.4) - f(0.6, -0.4)) < 1e-9
    fx, fy = lin.gradient(f, 0.6, -0.4)
    # plane reproduces the first-order change
    assert abs((P(0.7, -0.4) - P(0.6, -0.4)) - fx * 0.1) < 1e-9
    assert abs((P(0.6, -0.3) - P(0.6, -0.4)) - fy * 0.1) < 1e-9
