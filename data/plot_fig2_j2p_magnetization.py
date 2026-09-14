from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from plot_tm_spectrum_j2p import branch, load_dataset


HERE = Path(__file__).resolve().parent
POINTS_CSV = HERE / "fig2_j2p_magnetization_points.csv"
SUMMARY_CSV = HERE / "fig2_j2p_magnetization_summary.csv"
OUT_STEM = HERE / "fig2_j2p_magnetization"

# Where the panel-(a) guide line is drawn down to zero. None means "use the
# fitted crossing", which is what we want: the line then reaches zero exactly
# where the dashed marker sits, so the picture cannot disagree with the fit.
# The cost is a steep descent, because the crossing lies only 0.0038 past the
# last ordered point - that steepness is honest, since it is where the data put
# the transition. Setting a number instead buys a gentler curve at the price of
# the line and the dashed marker no longer coinciding. Either way the vertex is
# interpolation for the eye: it adds no data point, and every drawn marker still
# corresponds to one fitted intercept.
GUIDE_ZERO_AT: float | None = None

# One marker/colour per D, shared with figure 4's energy panel so a given bond
# dimension never appears as two different symbols across the paper.
D_STYLE = {
    4: {"marker": "v", "color": "#4E79A7", "label": "$D=4$"},
    5: {"marker": "<", "color": "#76B7B2", "label": "$D=5$"},
    6: {"marker": "s", "color": "#E6862A", "label": "$D=6$"},
    7: {"marker": "^", "color": "#59A14F", "label": "$D=7$"},
    8: {"marker": "D", "color": "#7F6D9D", "label": "$D=8$"},
    10: {"marker": "o", "color": "#C0392B", "label": "$D=10$"},
}


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def load_points() -> list[dict]:
    rows: list[dict] = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "j2p": float(row["j2p"]),
                    "D": int(row["D"]),
                    "chi": int(row["chi"]),
                    "M2": as_float(row["M2"]),
                    "M": as_float(row["M"]),
                    "xi": as_float(row["xi"]),
                    "inv_xi": as_float(row["inv_xi"]),
                    "energy": as_float(row["energy"]),
                    "status": row["status"],
                    "fit_used": row["fit_used"].lower() == "true",
                }
            )
    return rows


def load_summary() -> list[dict]:
    rows: list[dict] = []
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m2 = as_float(row["M2_intercept"])
            rows.append(
                {
                    "j2p": float(row["j2p"]),
                    "n_fit": int(row["n_fit_xi_ge_1"]),
                    "M2_intercept": m2,
                    "M0": math.sqrt(m2) if m2 > 0 else 0.0,
                    "slope": as_float(row["slope"]),
                    "r2": as_float(row["r2"]),
                }
            )
    return rows


def zero_crossing(summary: list[dict]) -> float:
    rows = sorted(summary, key=lambda r: r["j2p"])
    for left, right in zip(rows, rows[1:]):
        y0 = left["M2_intercept"]
        y1 = right["M2_intercept"]
        if y0 == 0:
            return left["j2p"]
        if y0 * y1 < 0:
            x0 = left["j2p"]
            x1 = right["j2p"]
            return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return math.nan


def plot_left(ax: plt.Axes, summary: list[dict]) -> None:
    ax.set_box_aspect(1)
    rows = sorted(summary, key=lambda r: r["j2p"])
    x = np.array([r["j2p"] for r in rows])
    y = np.array([r["M0"] for r in rows])
    positive = np.array([r["M2_intercept"] > 0 for r in rows])

    jc = zero_crossing(rows)

    # The guide line gets an extra vertex at (jc, 0) that is NOT drawn as a
    # marker. Without it the line runs straight from the last ordered point at
    # J2p=0.40 to the first disordered one at 0.42, so it meets zero at 0.42
    # while the dashed line marks the fitted transition at jc - a visible kink
    # that also disagrees with the fit. The vertex is interpolation for the eye
    # only: it adds no data point, and every plotted marker still corresponds to
    # one fitted intercept.
    vertex = GUIDE_ZERO_AT if GUIDE_ZERO_AT is not None else jc
    line_x, line_y = x, y
    if not math.isnan(vertex):
        pos = int(np.searchsorted(x, vertex))
        line_x = np.insert(x, pos, vertex)
        line_y = np.insert(y, pos, 0.0)

    ax.plot(line_x, line_y, color="#3B6EA8", linewidth=1.05, zorder=1)
    ax.scatter(x[positive], y[positive], s=14, color="#3B6EA8", edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter(
        x[~positive],
        y[~positive],
        s=14,
        facecolor="white",
        edgecolor="#3B6EA8",
        linewidth=0.75,
        zorder=3,
    )

    if not math.isnan(jc):
        ax.axvline(jc, color="#9B1D20", linestyle="--", linewidth=0.75, alpha=0.8)

    ax.set_xlabel(r"$J'_2/J_1$")
    ax.set_ylabel(r"$m_0$")
    ax.set_xlim(0.295, 0.505)
    ax.set_ylim(-0.005, 0.158)
    ax.grid(alpha=0.25)
    ax.text(-0.115, 1.02, "(a)", transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom", ha="left")


def plot_right(
    ax: plt.Axes,
    colorbar_ax: plt.Axes,
    points: list[dict],
    summary: list[dict],
    highlight_j2p: float = 0.40,
) -> None:
    ax.set_box_aspect(1)
    fit_points = [r for r in points if r["fit_used"] and r["xi"] >= 1.0]
    j2ps = sorted({r["j2p"] for r in fit_points})
    cmap = plt.get_cmap("viridis_r")
    norm = Normalize(vmin=min(j2ps), vmax=max(j2ps))
    fit_by_j2p = {round(r["j2p"], 10): r for r in summary}

    for j2p in j2ps:
        rows = [r for r in fit_points if abs(r["j2p"] - j2p) < 1e-9]
        fit = fit_by_j2p[round(j2p, 10)]
        color = cmap(norm(j2p))
        ax.scatter(
            [r["inv_xi"] for r in rows],
            [r["M2"] for r in rows],
            s=7,
            color=color,
            edgecolor="none",
            alpha=0.95,
            zorder=3,
        )
        xmax_j = max(r["inv_xi"] for r in rows) * 1.05
        xs = np.linspace(0.0, xmax_j, 120)
        ys = fit["slope"] * xs + fit["M2_intercept"]
        is_highlight = abs(j2p - highlight_j2p) < 1e-9
        ax.plot(
            xs,
            ys,
            color="black" if is_highlight else color,
            linestyle=(0, (3.2, 2.0)),
            linewidth=0.85 if is_highlight else 0.45,
            alpha=0.95 if is_highlight else 0.75,
            zorder=4 if is_highlight else 2,
        )
        ax.scatter(
            [0.0],
            [fit["M2_intercept"]],
            marker="*",
            s=26 if is_highlight else 18,
            color=color,
            edgecolor="black" if is_highlight else "none",
            linewidth=0.35 if is_highlight else 0.0,
            zorder=5,
        )
        if is_highlight:
            label_x = 0.22
            label_y = fit["slope"] * label_x + fit["M2_intercept"] + 0.003
            ax.text(
                label_x,
                label_y,
                rf"$J'_2/J_1={highlight_j2p:.1f}$",
                ha="left",
                va="bottom",
                fontsize=5.4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
                zorder=6,
            )

    ax.axhline(0.0, color="#777777", linewidth=0.55)
    ax.set_xlabel(r"$1/\xi$")
    ax.set_ylabel(r"$m^2$")
    ax.set_xlim(-0.02, max(r["inv_xi"] for r in fit_points) * 1.05)
    y_min = min([r["M2_intercept"] for r in summary] + [r["M2"] for r in fit_points])
    y_max = max(r["M2"] for r in fit_points)
    ax.set_ylim(min(-0.005, y_min * 1.12), y_max * 1.15)
    ax.grid(alpha=0.25)
    ax.text(-0.115, 1.02, "(b)", transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom", ha="left")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=colorbar_ax)
    cbar.ax.set_xlabel(r"$J'_2/J_1$", labelpad=4)
    cbar.set_ticks([0.30, 0.35, 0.40, 0.45, 0.50])
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.invert_yaxis()


def plot_spectrum(ax: plt.Axes, j2p: float = 0.50) -> None:
    """Transfer-matrix excitation spectrum, panel (c).

    This panel used to live in figure 4. It sits here because the main text
    discusses it (onset of finite wave-vector correlations) well before the
    energy-versus-inverse-correlation-length scaling that took its place
    in figure 4, so the panel
    order now follows the order in which the panels are referenced.
    """
    ax.set_box_aspect(1)
    rows = load_dataset()
    low_color = "#3B6EA8"
    high_color = "#788690"
    min_color = "#B23A48"

    for band in range(10, 0, -1):
        points = branch(rows, j2p, band)
        x = np.array([float(point["k_over_pi"]) for point in points])
        y = np.array([float(point["gap"]) for point in points])
        is_lowest = band == 1
        color = low_color if is_lowest else high_color
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.15 if is_lowest else 0.5,
            alpha=1.0 if is_lowest else 0.42,
            zorder=2 if is_lowest else 1,
        )
        ax.scatter(
            x,
            y,
            s=7 if is_lowest else 4,
            color=color,
            edgecolor="none",
            alpha=1.0 if is_lowest else 0.46,
            zorder=3,
        )

    lowest = branch(rows, j2p, 1)
    minimum = min(lowest, key=lambda point: float(point["gap"]))
    ax.axvline(2.0 / 3.0, color="#777777", linestyle=":", linewidth=0.6, alpha=0.65)
    ax.scatter(
        [minimum["k_over_pi"]],
        [minimum["gap"]],
        marker="*",
        s=32,
        color=min_color,
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
    )
    ax.set_title(rf"$J'_2/J_1={j2p:.1f}$, $D=8$", pad=3)
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(0.25, 1.56)
    ax.set_xticks([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    ax.set_xticklabels([r"$0(\Gamma)$", r"$\pi/3$", r"$2\pi/3(K)$", r"$\pi$"])
    ax.set_yticks([0.4, 0.8, 1.2, 1.6])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\Delta_n(k)=-\ln|\lambda_n(k)|$")
    ax.grid(alpha=0.25)
    ax.text(-0.115, 1.02, "(c)", transform=ax.transAxes, fontsize=8.0,
            fontweight="bold", va="bottom", ha="left")


def main() -> None:
    points = load_points()
    summary = load_summary()

    plt.rcParams.update(
        {
            "font.size": 6.2,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.0, 2.18), dpi=220)
    grid = fig.add_gridspec(
        1,
        7,
        # Keep the colorbar close to panel (b), with extra room before panel (c)
        # for the right-side colorbar labels and panel-(c) y-axis labels.
        width_ratios=[1.0, 0.46, 1.0, 0.035, 0.045, 0.58, 1.0],
        left=0.055,
        right=0.98,
        bottom=0.18,
        top=0.88,
        wspace=0.0,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 6]),
    ]
    colorbar_ax = fig.add_subplot(grid[0, 4])
    plot_left(axes[0], summary)
    plot_right(axes[1], colorbar_ax, points, summary)
    plot_spectrum(axes[2])
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(OUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, **save_kwargs)
    print(OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
