from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Rectangle


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "ipeps_merge_scheme"

J1_COLOR = "#355F78"
J2_COLOR = "#C85A3A"
CELL_COLOR = "#398A83"
VIRTUAL_COLOR = "#878C94"
A_COLOR = "#F7F7F4"
B_COLOR = "#E6A45D"
TENSOR_COLOR = "#EEE8D8"


def honeycomb_site(i: int, j: int, sublattice: str) -> tuple[float, float]:
    """Coordinates used by the merge geometry in TeneT.jl."""
    x_b = math.sqrt(3.0) * j + 0.5 * math.sqrt(3.0) * i
    y_b = -1.5 * i
    if sublattice == "B":
        return x_b, y_b
    return x_b + 0.5 * math.sqrt(3.0), y_b - 0.5


def draw_honeycomb(ax: plt.Axes) -> None:
    cells = [(i, j) for i in range(3) for j in range(3)]

    # The shaded two-site primitive cells are the objects merged into tensors.
    for i, j in cells:
        b = honeycomb_site(i, j, "B")
        a = honeycomb_site(i, j, "A")
        center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        ax.add_patch(
            Ellipse(
                center,
                width=1.26,
                height=0.64,
                angle=-30.0,
                facecolor=CELL_COLOR,
                edgecolor=CELL_COLOR,
                alpha=0.11,
                linewidth=1.0,
                zorder=0,
            )
        )

    # Nearest-neighbor bonds: intracell plus right/down intercell terms.
    for i, j in cells:
        a = honeycomb_site(i, j, "A")
        b = honeycomb_site(i, j, "B")
        ax.plot([b[0], a[0]], [b[1], a[1]], color=J1_COLOR, lw=1.55, zorder=1)
        if j + 1 < 3:
            b_right = honeycomb_site(i, j + 1, "B")
            ax.plot([a[0], b_right[0]], [a[1], b_right[1]], color=J1_COLOR, lw=1.55, zorder=1)
        if i + 1 < 3:
            b_down = honeycomb_site(i + 1, j, "B")
            ax.plot([a[0], b_down[0]], [a[1], b_down[1]], color=J1_COLOR, lw=1.55, zorder=1)

    # Sublattice-selective next-nearest-neighbor bonds.
    for i, j in cells:
        b = honeycomb_site(i, j, "B")
        for di, dj in ((0, 1), (1, 0), (1, -1)):
            neighbor = (i + di, j + dj)
            if neighbor in cells:
                b2 = honeycomb_site(*neighbor, "B")
                ax.plot(
                    [b[0], b2[0]],
                    [b[1], b2[1]],
                    color=J2_COLOR,
                    lw=1.25,
                    ls=(0, (3.0, 2.0)),
                    zorder=0.8,
                )

    for i, j in cells:
        for sublattice, fill in (("B", B_COLOR), ("A", A_COLOR)):
            x, y = honeycomb_site(i, j, sublattice)
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=0.115,
                    facecolor=fill,
                    edgecolor="#252525",
                    linewidth=0.85,
                    zorder=3,
                )
            )

    b0 = honeycomb_site(0, 0, "B")
    a0 = honeycomb_site(0, 0, "A")
    ax.text(b0[0] - 0.05, b0[1] + 0.25, r"$B$", ha="center", va="bottom", fontsize=8)
    ax.text(a0[0] + 0.02, a0[1] - 0.25, r"$A$", ha="center", va="top", fontsize=8)
    ax.annotate(
        "two-site cell",
        xy=((a0[0] + b0[0]) / 2.0, (a0[1] + b0[1]) / 2.0),
        xytext=(1.58, 0.58),
        fontsize=7.3,
        color=CELL_COLOR,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": CELL_COLOR, "lw": 0.8},
    )

    ax.set_xlim(-0.38, 5.78)
    ax.set_ylim(-3.72, 0.92)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("honeycomb lattice", fontsize=9, pad=1.5)


def tensor_sites(center: tuple[float, float]) -> dict[str, tuple[float, float]]:
    x, y = center
    return {"B": (x - 0.105, y + 0.105), "A": (x + 0.105, y - 0.105)}


def draw_square_mapping(ax: plt.Axes) -> None:
    centers = {
        "r": (0.0, 0.75),
        "x": (1.32, 0.75),
        "y": (0.0, -0.57),
        "xy": (1.32, -0.57),
    }

    # Virtual square-lattice links, including dangling bonds indicating infinity.
    for p, q in (("r", "x"), ("r", "y"), ("x", "xy"), ("y", "xy")):
        x1, y1 = centers[p]
        x2, y2 = centers[q]
        ax.plot([x1, x2], [y1, y2], color=VIRTUAL_COLOR, lw=3.0, zorder=0)
    for x, y in centers.values():
        for dx, dy in ((-0.48, 0), (0.48, 0), (0, -0.48), (0, 0.48)):
            if -0.25 < x + dx < 1.57 and -0.82 < y + dy < 1.0:
                continue
            ax.plot([x, x + dx], [y, y + dy], color=VIRTUAL_COLOR, lw=3.0, zorder=0)

    size = 0.48
    for key, center in centers.items():
        x, y = center
        ax.add_patch(
            Rectangle(
                (x - size / 2.0, y - size / 2.0),
                size,
                size,
                facecolor=TENSOR_COLOR,
                edgecolor="#333333",
                linewidth=1.05,
                zorder=1,
            )
        )
        sites = tensor_sites(center)
        for sublattice, fill in (("B", B_COLOR), ("A", A_COLOR)):
            sx, sy = sites[sublattice]
            ax.add_patch(
                Circle(
                    (sx, sy),
                    radius=0.065,
                    facecolor=fill,
                    edgecolor="#252525",
                    linewidth=0.65,
                    zorder=4,
                )
            )

    sites = {key: tensor_sites(center) for key, center in centers.items()}

    # The six Hamiltonian contributions anchored to one merged unit cell.
    j1_pairs = (
        (sites["r"]["B"], sites["r"]["A"]),
        (sites["r"]["A"], sites["x"]["B"]),
        (sites["r"]["A"], sites["y"]["B"]),
    )
    for p, q in j1_pairs:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=J1_COLOR, lw=1.65, zorder=3)

    j2_pairs = (
        (sites["r"]["B"], sites["x"]["B"]),
        (sites["r"]["B"], sites["y"]["B"]),
        (sites["x"]["B"], sites["y"]["B"]),
    )
    for p, q in j2_pairs:
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            color=J2_COLOR,
            lw=1.35,
            ls=(0, (3.0, 2.0)),
            zorder=2.5,
        )

    # One physical leg is shown explicitly; the other nodes are translations.
    x0, y0 = centers["r"]
    ax.plot([x0, x0], [y0 + size / 2.0, y0 + 0.66], color=CELL_COLOR, lw=1.8, zorder=0.5)
    ax.text(
        x0 + 0.10,
        1.23,
        r"$s=(s_A,s_B),\quad d_{\rm phys}=4$",
        fontsize=7.3,
        ha="left",
        va="center",
    )
    ax.text(1.32, -0.91, r"$\mathcal{A}^{s_A s_B}_{lrud}$", fontsize=8.3, ha="center", va="top")
    ax.text(0.66, -0.48, r"$D$", fontsize=7.5, color=VIRTUAL_COLOR, ha="center", va="bottom")

    ax.set_xlim(-0.66, 1.98)
    ax.set_ylim(-1.35, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("effective square lattice", fontsize=9, pad=1.5)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.05), gridspec_kw={"width_ratios": [1.45, 1.0]})
    draw_honeycomb(axes[0])
    draw_square_mapping(axes[1])

    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(-0.02, 1.01, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")

    fig.text(0.542, 0.56, r"$\Longrightarrow$", fontsize=18, color="#333333", ha="center", va="center")
    fig.text(0.542, 0.64, "merge", fontsize=7.5, color=CELL_COLOR, ha="center", va="center")

    handles = [
        Line2D([0], [0], color=J1_COLOR, lw=1.7, label=r"$J_1$"),
        Line2D([0], [0], color=J2_COLOR, lw=1.4, ls=(0, (3.0, 2.0)), label=r"$J'_2$ on $B$"),
        Line2D([0], [0], color=CELL_COLOR, lw=5.0, alpha=0.28, label="merged cell"),
        Line2D([0], [0], color=VIRTUAL_COLOR, lw=3.0, label="virtual bond"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        fontsize=7.7,
        handlelength=2.0,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.18, wspace=0.22)

    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
