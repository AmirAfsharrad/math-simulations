"""2x2 linear maps: the math behind the ``matrix-2x2`` browser demo.

A 2x2 matrix ``A`` acts on the plane by ``x -> A x``. This module is the
authoritative (Python) implementation of the quantities the demo visualizes:

* the image of the standard basis vectors, which are exactly the columns of A;
* the determinant (the signed area-scaling factor) and the orientation;
* the standard example matrices (rotation, shear, scaling, reflection, ...);
* the image of an integer grid under A.

The browser demo (``../matrix-2x2/index.html``) mirrors these in a few lines of
JavaScript. Keep the two in sync: this file is the source of truth.
"""
from __future__ import annotations

import numpy as np


def as_matrix(a, b, c, d):
    """Build A = [[a, b], [c, d]]."""
    return np.array([[a, b], [c, d]], dtype=float)


def apply(A, v):
    """Image ``A v`` of a vector ``v`` (or a 2xk stack of column vectors)."""
    return np.asarray(A, dtype=float) @ np.asarray(v, dtype=float)


def columns(A):
    """Images of e1 and e2, i.e. the two columns of A: (A e1, A e2)."""
    A = np.asarray(A, dtype=float)
    return A[:, 0], A[:, 1]


def determinant(A):
    """Signed area-scaling factor of A."""
    A = np.asarray(A, dtype=float)
    return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])


def orientation(A, tol=1e-9):
    """'preserved' (det>0), 'flipped' (det<0), or 'collapsed' (det~0)."""
    d = determinant(A)
    if abs(d) < tol:
        return "collapsed"
    return "preserved" if d > 0 else "flipped"


def is_invertible(A, tol=1e-9):
    """True iff A has a (numerically) non-zero determinant."""
    return abs(determinant(A)) > tol


def rows(A):
    """The rows of A as vectors: (row_1, row_2, ...). y_i = row_i . x."""
    A = np.asarray(A, dtype=float)
    return tuple(A[i, :] for i in range(A.shape[0]))


def column_mixture(A, x):
    """The terms of y = Ax read as a mixture of columns: (x_1 a_1, x_2 a_2, ...).

    Summing them gives y; this is the 'output as a combination of columns' view.
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    return tuple(x[j] * A[:, j] for j in range(A.shape[1]))


def solve(A, y, tol=1e-9):
    """Solve A x = y. Returns the unique x, or None if A is singular.

    The row view: each equation row_i . x = y_i is a hyperplane; the solution is
    their intersection. Unique exactly when A is invertible.
    """
    A = np.asarray(A, dtype=float)
    if not is_invertible(A, tol):
        return None
    return np.linalg.solve(A, np.asarray(y, dtype=float))


# --- standard examples ------------------------------------------------------
IDENTITY = as_matrix(1, 0, 0, 1)


def rotation(deg):
    """Counter-clockwise rotation by ``deg`` degrees."""
    t = np.deg2rad(deg)
    return as_matrix(np.cos(t), -np.sin(t), np.sin(t), np.cos(t))


def scaling(sx, sy):
    """Axis-aligned scaling by sx (x) and sy (y)."""
    return as_matrix(sx, 0, 0, sy)


def shear(k, axis="x"):
    """Shear by factor k along the x-axis (default) or y-axis."""
    return as_matrix(1, k, 0, 1) if axis == "x" else as_matrix(1, 0, k, 1)


def reflection_yx():
    """Reflection across the line y = x."""
    return as_matrix(0, 1, 1, 0)


def projection_x():
    """Orthogonal projection onto the x-axis (singular)."""
    return as_matrix(1, 0, 0, 0)


PRESETS = {
    "identity": IDENTITY,
    "rotate30": rotation(30),
    "shear": shear(1),
    "scale": scaling(1.8, 0.5),
    "reflect_yx": reflection_yx(),
    "project_x": projection_x(),
}


def transformed_grid(A, n=4, step=1.0):
    """Integer grid lines on [-n, n]^2 mapped through A.

    Returns a list of ``(p0, p1)`` endpoint pairs, one per vertical and
    horizontal grid line. A is linear, so straight lines map to straight lines
    and two endpoints per line are enough.
    """
    cs = np.arange(-n, n + step / 2, step)
    lines = []
    for c in cs:
        lines.append((apply(A, [c, -n]), apply(A, [c, n])))   # vertical x = c
        lines.append((apply(A, [-n, c]), apply(A, [n, c])))   # horizontal y = c
    return lines
