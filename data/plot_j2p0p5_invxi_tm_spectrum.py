from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FormatStrFormatter

from plot_fig2_j2p_magnetization import D_STYLE
from plot_fig2_j2p_magnetization import load_points as load_magnetization_points
from plot_j2p0p5_invxi_vs_invD import fit, load_points


HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "j2p0p5_invxi_energy"


def plot_scaling(ax: plt.Axes) -> None:
    rows = load_points()
    result = fit(rows)
    # One marker/colour per D, shared with figure 2 so the same bond dimension
    # never appears as two different symbols across the paper.
    colors = {4: "#4E79A7", 5: "#76B7B2", 6: "#E6862A",
              7: "#59A14F", 8: "#7F6D9D", 10: "#C0392B"}
    markers = {4: "v", 5: "<", 6: "s", 7: "^", 8: "D", 10: "o"}

    for row in rows:
        D = int(row["D"])
        ax.scatter(
            [row["inv_D"]],
            [row["inv_xi"]],
            marker=markers[D],
            s=24,
            color=colors[D],
            edgecolor="white",
            linewidth=0.4,
            label=rf"$D={D}$",
            zorder=4,
        )

    x_line = np.linspace(0.0, 0.26, 240)
    ax.plot(
        x_line,
        result["slope"] * x_line + result["intercept"],
        color="black",
        linestyle=(0, (3.2, 2.0)),
        linewidth=0.85,
        label="free fit",
        zorder=2,
    )
    ax.plot(
        x_line,
        result["origin_slope"] * x_line,
        color="#C43C39",
        linestyle=":",
        linewidth=1.0,
        label="zero-intercept fit",
        zorder=2,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.5, alpha=0.75, zorder=1)
    ax.text(
        0.96,
        0.08,
        rf"$(1/\xi)_\infty={result['intercept']:.3f}\pm{result['intercept_se']:.3f}$"
        + "\n"
        + rf"$R^2={result['r2']:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.6,
    )
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$1/\xi$")
    ax.set_xlim(-0.008, 0.265)
    ax.set_ylim(-0.12, 1.16)
    ax.grid(alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    # Order by D where the label says so, keeping any fit lines at the end,
    # rather than by a fixed index list that breaks when points are added.
    def _rank(pair):
        label = pair[1]
        if label.startswith("$D="):
            return (0, int(label.split("=")[1].rstrip("$")))
        return (1, 0)
    paired = sorted(zip(handles, labels), key=_rank)
    ax.legend(
        [h for h, _ in paired],
        [l for _, l in paired],
        loc="upper left",
        frameon=False,
        ncol=2,
        columnspacing=0.6,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def plot_energy(ax: plt.Axes, points: list[dict], target_j2p: float = 0.50) -> dict:
    ax.set_box_aspect(1)
    # This panel selects on its own marker, energy_fit_point, and NOT on fit_used.
    # The two panels answer different questions from partly different data: the
    # magnetization fit needs M at many xi, while the energy fit needs one
    # well-converged energy per D and can use bond dimensions for which no
    # magnetization was ever recorded (D=4 and D=5 here). Tying them to one flag
    # forced a compromise on both.
    #
    # The marker is applied to one state per D at that D's largest converged chi,
    # which is the same rule panel (a) beside it uses - the two panels of this
    # figure now extrapolate the same states, which they previously did not.
    target_rows = [
        row
        for row in points
        if "energy_fit_point" in row["status"]
        and abs(row["j2p"] - target_j2p) < 1e-9
    ]
    # Sort by D, or the legend follows CSV row order - which put D=6 last
    # once its row was appended after the others.
    target_rows.sort(key=lambda row: row["D"])
    missing = [(row["D"], row["chi"]) for row in target_rows if not math.isfinite(row["energy"])]
    if missing:
        raise ValueError(f"Missing energy for J2p={target_j2p} magnetization-fit points: {missing}")
    if len(target_rows) < 3:
        raise ValueError(f"Need at least three J2p={target_j2p} magnetization-fit points")

    x = np.array([1.0 / row["xi"] ** 3 for row in target_rows], dtype=float)
    y = np.array([row["energy"] for row in target_rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    for row, inv_xi3 in zip(target_rows, x):
        style = D_STYLE[row["D"]]
        ax.scatter(
            [inv_xi3],
            [row["energy"]],
            marker=style["marker"],
            s=24,
            color=style["color"],
            edgecolor="white",
            linewidth=0.4,
            label=style["label"],
            zorder=4,
        )

    xmax = max(x) * 1.08
    xs = np.linspace(0.0, xmax, 160)
    ax.plot(
        xs,
        slope * xs + intercept,
        color="black",
        linestyle=(0, (3.2, 2.0)),
        linewidth=0.9,
        zorder=2,
    )
    ax.scatter(
        [0.0],
        [intercept],
        marker="*",
        s=34,
        color="#2A9D8F",
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )

    ax.text(
        0.96,
        0.05,
        rf"$e_\infty={intercept:.6f}$" + "\n" + rf"$R^2={r2:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.6,
    )
    ax.set_xlabel(r"$1/\xi^3$")
    ax.set_ylabel(r"$e$")
    ax.set_xlim(-0.008, xmax)
    energies = list(y) + [intercept]
    y_span = max(energies) - min(energies)
    ax.set_ylim(min(energies) - 0.08 * y_span, max(energies) + 0.12 * y_span)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax.grid(alpha=0.22)
    # One legend entry per D, not per point: a given D can contribute several
    # chi values here (D=10 supplies both chi512 and chi768).
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    ax.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="upper left",
        frameon=False,
        handlelength=1.2,
        labelspacing=0.3,
    )

    return {
        "n_fit": len(target_rows),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
    }


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 5.6,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.1,
            "ytick.labelsize": 5.1,
            "legend.fontsize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.82), dpi=220)
    for ax in axes:
        ax.set_box_aspect(1)

    plot_scaling(axes[0])
    energy_fit = plot_energy(axes[1], load_magnetization_points())
    for ax, label in zip(axes, ("(a)", "(b)"), strict=True):
        ax.text(
            -0.17,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    fig.tight_layout(w_pad=0.6, pad=0.18)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(OUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, **save_kwargs)
    plt.close(fig)
    print(
        "J2p=0.5 energy fit: "
        f"n={energy_fit['n_fit']}, e_inf={energy_fit['intercept']:.12f}, "
        f"slope={energy_fit['slope']:.12f}, R2={energy_fit['r2']:.9f}"
    )
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
