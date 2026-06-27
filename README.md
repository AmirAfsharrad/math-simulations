# math-simulations

Interactive math visualizations for [EE263](https://ee263.stanford.edu)
(Matrix Methods: Singular Value Decomposition, Stanford).

Each simulation has two parts:

- an **authoritative Python implementation** of the underlying math, in
  [`python/`](python/), reusable for slides, homework, and in-class scripts, and
  covered by tests in [`tests/`](tests/); and
- a small, self-contained **browser demo** (HTML + a few lines of JavaScript)
  that mirrors that math so students can change parameters and see the effect.

The Python is the source of truth. The browser demos intentionally re-implement
the same few formulas in JS so they can run client-side with no build step.

This repository is embedded into my website as a git submodule at
`/teaching/ee263-matrix-methods/sim/`, so the demos are served at
<https://amirafsharrad.github.io/teaching/ee263-matrix-methods/sim/>.

## Simulations

| Demo | What it shows | Math |
|------|----------------|------|
| [`matrix-2x2/`](matrix-2x2/) | A 2x2 matrix as a transformation of the plane | [`python/linmaps.py`](python/linmaps.py) |
| [`linearization/`](linearization/) | Tangent line (1D) and tangent plane (2D) | [`python/linearization.py`](python/linearization.py) |

## Python

```
python/    authoritative implementations (NumPy)
tests/     pytest tests
```

Run the tests:

```bash
pip install numpy pytest
pytest
```

## Adding a simulation

1. Implement the math in `python/<name>.py`, with tests in `tests/`.
2. Build a self-contained `<name>/index.html` demo that mirrors it.
3. Add a card to `index.html`.
4. In the website repo, update the submodule pointer.

## License

MIT (see `LICENSE`).
