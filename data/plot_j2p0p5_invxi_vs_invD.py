from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
POINTS_CSV = HERE / "j2p0p5_xi_vs_D_points.csv"
SUMMARY_CSV = HERE / "j2p0p5_invxi_vs_invD_summary.csv"
OUT_STEM = HERE / "j2p0p5_invxi_vs_invD"
# The originally published set D=4,5,6,8, with D=10 and D=7 added. One state per
# D, each at that D's largest converged chi. D=7 is included so that this panel
# and the energy panel beside it extrapolate the same bond dimensions.
#
# This extrapolation is NOT robust against the choice of D, and the code should
# say so where the choice is made. The unconstrained intercept reads:
#     D=4,5,6,8      -0.0896 +/- 0.0468   R2=0.994   negative, 1.9 sigma
#     D=4,5,6,8,10   -0.0061 +/- 0.0556   R2=0.984   zero to within 0.1 sigma
#     D=4,5,6,7,8,10 -0.0347 +/- 0.0643   R2=0.971   negative, 0.5 sigma
#     D=6,7,8,10     +0.1307 +/- 0.1059   R2=0.885   POSITIVE, 1.2 sigma
# so its SIGN depends on which bond dimensions enter. The cause is visible in
# the data: 1/xi falls by 0.137 from D=6 to D=7 but only by 0.036 from D=8 to
# D=10, so the curve is flattening, and a straight line through the large-D end
# alone extrapolates upward. Any claim drawn from the sign of this intercept has
# to be stated with that caveat.
#
# Recompute this table from the CSV after ANY point changes - the D=8 chi1024
# re-optimisation of 2026-09-03 silently invalidated every row above.
SELECTED_D = (4, 5, 6, 7, 8, 10)


def load_points() -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["branch"] != "primary":
                continue
            D = int(row["D"])
            xi = float(row["xi"])
            rows.append(
                {
                    "D": D,
                    "chi": int(row["chi"]),
                    "xi": xi,
                    "inv_D": 1.0 / D,
                    "inv_xi": 1.0 / xi,
                    "energy": float(row["energy"]),
                    "frontier": row["frontier"].lower() == "true",
                    "source": row["source"],
                }
            )
    selected = [row for row in rows if row["frontier"] and row["D"] in SELECTED_D]
    assert tuple(sorted(int(row["D"]) for row in selected)) == SELECTED_D
    return sorted(selected, key=lambda row: int(row["D"]))


def fit(rows: list[dict[str, float | int | str | bool]]) -> dict[str, float]:
    x = np.array([float(row["inv_D"]) for row in rows])
    y = np.array([float(row["inv_xi"]) for row in rows])
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([slope, intercept])
    ss_res = float(residual @ residual)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    covariance = (ss_res / (len(x) - 2)) * np.linalg.inv(design.T @ design)

    origin_slope = float(np.dot(x, y) / np.dot(x, x))
    origin_residual = y - origin_slope * x
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "intercept_se": float(np.sqrt(covariance[1, 1])),
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "origin_slope": origin_slope,
        "origin_rmse": float(np.sqrt(np.mean(origin_residual**2))),
    }


def write_summary(
    rows: list[dict[str, float | int | str | bool]], result: dict[str, float]
) -> None:
    columns = [
        "j2p",
        "n_fit",
        "Ds",
        "chis",
        "slope",
        "intercept",
        "intercept_se",
        "r2",
        "rmse",
        "origin_slope",
        "origin_rmse",
    ]
    record = {
        "j2p": 0.5,
        "n_fit": len(rows),
        "Ds": ",".join(str(row["D"]) for row in rows),
        "chis": ",".join(str(row["chi"]) for row in rows),
        **result,
    }
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(record)


def plot(
    rows: list[dict[str, float | int | str | bool]], result: dict[str, float]
) -> None:
    x = np.array([float(row["inv_D"]) for row in rows])
    y = np.array([float(row["inv_xi"]) for row in rows])
    # One marker/colour per D, shared with figure 2 so the same bond dimension
    # never appears as two different symbols across the paper.
    colors = {4: "#4E79A7", 5: "#76B7B2", 6: "#E6862A",
              7: "#59A14F", 8: "#7F6D9D", 10: "#C0392B"}
    markers = {4: "v", 5: "<", 6: "s", 7: "^", 8: "D", 10: "o"}

    fig, ax = plt.subplots(figsize=(3.35, 3.05), dpi=220)
    ax.set_box_aspect(1)
    for row in rows:
        D = int(row["D"])
        ax.scatter(
            [row["inv_D"]],
            [row["inv_xi"]],
            marker=markers[D],
            s=28,
            color=colors[D],
            edgecolor="white",
            linewidth=0.45,
            label=rf"$D={D}$",
            zorder=4,
        )

    x_line = np.linspace(0.0, 0.26, 240)
    ax.plot(
        x_line,
        result["slope"] * x_line + result["intercept"],
        color="black",
        linestyle=(0, (3.2, 2.0)),
        linewidth=0.9,
        label="free linear fit",
        zorder=2,
    )
    ax.plot(
        x_line,
        result["origin_slope"] * x_line,
        color="#C43C39",
        linestyle=":",
        linewidth=1.05,
        label="fit constrained to zero",
        zorder=2,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.55, alpha=0.75, zorder=1)

    ax.text(
        0.965,
        0.125,
        rf"$(1/\xi)_\infty={result['intercept']:.3f}\pm{result['intercept_se']:.3f}$"
        + "\n"
        + rf"$R^2={result['r2']:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
    )
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$1/\xi$")
    ax.set_xlim(-0.008, 0.265)
    ax.set_ylim(-0.12, 1.16)
    ax.grid(alpha=0.24)
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
        columnspacing=0.9,
        handlelength=1.7,
    )
    ax.set_title(r"$J'_2/J_1=0.5$", pad=4)

    fig.tight_layout(pad=0.25)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(OUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, **save_kwargs)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 6.2,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = load_points()
    result = fit(rows)
    write_summary(rows, result)
    plot(rows, result)

    print(f"fit points = {[(row['D'], row['chi'], row['xi']) for row in rows]}")
    print(f"intercept = {result['intercept']:+.9f} +/- {result['intercept_se']:.9f}")
    print(f"slope = {result['slope']:.9f}, R2 = {result['r2']:.6f}")
    print(f"origin slope = {result['origin_slope']:.9f}")
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
