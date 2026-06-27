import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import linmaps as lm  # noqa: E402


def test_columns_are_images_of_basis():
    A = lm.as_matrix(1, 2, 3, 4)
    c1, c2 = lm.columns(A)
    assert np.allclose(c1, lm.apply(A, [1, 0]))
    assert np.allclose(c2, lm.apply(A, [0, 1]))


def test_rotation_determinant_is_one():
    assert abs(lm.determinant(lm.rotation(37)) - 1.0) < 1e-12
    assert lm.orientation(lm.rotation(37)) == "preserved"


def test_scaling_determinant_is_product():
    assert abs(lm.determinant(lm.scaling(1.8, 0.5)) - 0.9) < 1e-12


def test_projection_is_singular():
    assert not lm.is_invertible(lm.projection_x())
    assert lm.orientation(lm.projection_x()) == "collapsed"


def test_reflection_flips_orientation():
    assert lm.orientation(lm.reflection_yx()) == "flipped"


def test_transformed_grid_count():
    lines = lm.transformed_grid(lm.IDENTITY, n=4, step=1.0)
    assert len(lines) == 2 * 9  # 9 positions (-4..4), vertical + horizontal


def test_column_mixture_sums_to_Ax():
    A = lm.as_matrix(1, -1, 2, 1)
    x = [1.0, -0.5]
    terms = lm.column_mixture(A, x)
    assert np.allclose(sum(terms), lm.apply(A, x))


def test_rows_give_outputs():
    A = lm.as_matrix(2, 1, -1, 1)
    x = np.array([1.0, 2.0])
    r1, r2 = lm.rows(A)
    y = lm.apply(A, x)
    assert abs(r1 @ x - y[0]) < 1e-12 and abs(r2 @ x - y[1]) < 1e-12


def test_solve_round_trip_and_singular():
    A = lm.as_matrix(2, 1, -1, 1)
    y = np.array([4.0, 1.0])
    x = lm.solve(A, y)
    assert np.allclose(lm.apply(A, x), y)         # x = (1, 2)
    assert lm.solve(lm.projection_x(), [0, 1]) is None  # singular -> None
