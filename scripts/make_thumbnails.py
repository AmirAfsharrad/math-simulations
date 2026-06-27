import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "thumbs")

BLUE = "#1f5fa6"
RED = "#c0392b"
GREEN = "#2e8b57"
PURPLE = "#6a3d9a"
INK = "#222222"
MUTED = "#9aa3ad"
GRID = "#e8edf2"


def setup_canvas():
    fig = plt.figure(figsize=(10, 3), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def draw_grid(ax, x0, y0, w, h, lim=2.5):
    for v in np.arange(-2, 3):
        x = x0 + (v + lim) / (2 * lim) * w
        y = y0 + (v + lim) / (2 * lim) * h
        ax.plot([x, x], [y0, y0 + h], color=GRID, lw=1)
        ax.plot([x0, x0 + w], [y, y], color=GRID, lw=1)
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color=MUTED, lw=1.2)
    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h], color=MUTED, lw=1.2)

    def to_panel(p):
        return x0 + (p[0] + lim) / (2 * lim) * w, y0 + (p[1] + lim) / (2 * lim) * h

    return to_panel


def arrow(ax, start, end, color, lw=2.4, alpha=1.0, style="-"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=style,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=13,
        ),
    )


def matrix_2x2():
    fig, ax = setup_canvas()
    A = np.array([[1.25, 0.75], [-0.45, 1.05]])
    center = np.array([5.0, 1.42])
    scale = 0.55

    for c in np.linspace(-2, 2, 9):
        pts1 = np.array([[c, -2.1], [c, 2.1]]) @ A.T
        pts2 = np.array([[-2.1, c], [2.1, c]]) @ A.T
        ax.plot(center[0] + scale * pts1[:, 0], center[1] + scale * pts1[:, 1], color=GRID, lw=1)
        ax.plot(center[0] + scale * pts2[:, 0], center[1] + scale * pts2[:, 1], color=GRID, lw=1)

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ A.T
    ax.fill(center[0] + scale * square[:, 0], center[1] + scale * square[:, 1], color=BLUE, alpha=0.10)
    ax.plot(center[0] + scale * square[:, 0], center[1] + scale * square[:, 1], color=BLUE, lw=1.7, alpha=0.65)

    origin = center
    a1 = center + scale * A[:, 0]
    a2 = center + scale * A[:, 1]
    arrow(ax, origin, a1, RED, 2.8)
    arrow(ax, origin, a2, GREEN, 2.8)
    ax.text(a1[0] + 0.08, a1[1] - 0.02, "$a_1$", color=RED, fontsize=13)
    ax.text(a2[0] + 0.06, a2[1] + 0.04, "$a_2$", color=GREEN, fontsize=13)
    fig.savefig(os.path.join(OUTDIR, "matrix-2x2.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def linearization():
    fig, ax = setup_canvas()
    xs = np.linspace(1.4, 8.6, 500)
    x0 = 4.45
    y = 0.13 * (xs - 5.1) ** 2 + 1.05
    y0 = 0.13 * (x0 - 5.1) ** 2 + 1.05
    slope = 2 * 0.13 * (x0 - 5.1)
    tangent = y0 + slope * (xs - x0)

    ax.plot(xs, y, color=BLUE, lw=3.2, solid_capstyle="round")
    ax.plot(xs, tangent, color=RED, lw=2.4, dashes=(6, 4), solid_capstyle="round")
    ax.scatter([x0], [y0], s=42, color=INK, zorder=4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    fig.savefig(os.path.join(OUTDIR, "linearization.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def linear_views():
    fig, ax = setup_canvas()

    left = draw_grid(ax, 0.9, 0.35, 3.5, 2.3, lim=2.4)
    x = np.array([0.55, 0.85])
    r1 = np.array([1.5, 0.75])
    r2 = np.array([-0.85, 1.2])
    y = np.array([r1 @ x, r2 @ x])
    for row, val, color in [(r1, y[0], BLUE), (r2, y[1], PURPLE)]:
        d = np.array([-row[1], row[0]])
        d = d / np.linalg.norm(d)
        p0 = row * val / (row @ row)
        for k in [-2, -1, 1, 2]:
            pp = p0 + k * 0.65 * row / np.linalg.norm(row)
            p1, p2 = pp - 3 * d, pp + 3 * d
            ax.plot(*zip(left(p1), left(p2)), color=color, lw=0.9, alpha=0.15)
        p1, p2 = p0 - 3 * d, p0 + 3 * d
        ax.plot(*zip(left(p1), left(p2)), color=color, lw=2.0)
    ax.scatter(*left(x), s=36, color=INK, zorder=5)
    ax.text(left(x)[0] + 0.08, left(x)[1] + 0.07, "$x$", color=INK, fontsize=13)

    right = draw_grid(ax, 5.6, 0.35, 3.5, 2.3, lim=3.2)
    a1 = np.array([1.7, -0.7])
    a2 = np.array([0.75, 1.25])
    xmix = np.array([0.95, 1.1])
    t1 = xmix[0] * a1
    yy = t1 + xmix[1] * a2
    o = np.array([0.0, 0.0])
    arrow(ax, right(o), right(a1), RED, 2.5)
    arrow(ax, right(o), right(a2), GREEN, 2.5)
    arrow(ax, right(o), right(t1), RED, 1.8, alpha=0.35, style="--")
    arrow(ax, right(t1), right(yy), GREEN, 1.8, alpha=0.35, style="--")
    arrow(ax, right(o), right(yy), INK, 2.4)
    ax.scatter(*right(yy), s=32, color=INK, zorder=5)
    ax.text(right(yy)[0] + 0.08, right(yy)[1] + 0.06, "$y$", color=INK, fontsize=13)

    fig.savefig(os.path.join(OUTDIR, "linear-views.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def linear_or_not():
    fig, ax = setup_canvas()
    left = draw_grid(ax, 1.0, 0.45, 2.9, 2.1, lim=2.3)
    right = draw_grid(ax, 6.1, 0.45, 2.9, 2.1, lim=2.3)
    x = np.array([1.0, 0.7])
    y = np.array([0.95, -0.55])
    arrow(ax, left([0, 0]), left(x), BLUE, 2.7)
    arrow(ax, right([0, 0]), right(y), RED, 2.7)
    ax.scatter(*left(x), s=34, color=BLUE, zorder=5)
    ax.scatter(*right(y), s=34, color=RED, zorder=5)
    ax.text(5.0, 1.55, "$f\\,?$", ha="center", va="center", color=INK, fontsize=25)
    ax.annotate("", xy=(5.75, 1.48), xytext=(4.25, 1.48),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.0, mutation_scale=14))
    fig.savefig(os.path.join(OUTDIR, "linear-or-not.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    matrix_2x2()
    linearization()
    linear_views()
    linear_or_not()


if __name__ == "__main__":
    main()
