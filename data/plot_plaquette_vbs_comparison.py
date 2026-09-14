from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
DATA_JSON = HERE / "plaquette_vbs_comparison_data.json"
ARTICLE_POINTS = HERE / "fig2_j2p_magnetization_points.csv"
OUT_STEM = HERE / "plaquette_vbs_comparison"

STRONG = "#2B6CB0"
WEAK = "#B9C0C8"
PLAQUETTE = "#D97824"
MERGE = "#2F7D66"
KDIMER = "#7D3C98"
TEXT = "#262626"
PAIR_COLORS = ("#EF3B2C", "#E6AC00", "#1599E5")
PATTERN = np.array([[1, 3, 5, 2, 4, 6], [2, 4, 6, 1, 3, 5]], dtype=int)


def load_data() -> dict:
    with DATA_JSON.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_merge_points(data: dict) -> list[dict]:
    points: list[dict] = list(data["merge_additional_points"])
    with ARTICLE_POINTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["j2p"]) - 0.5) > 1.0e-12:
                continue
            if row["fit_used"].lower() != "true" or row["energy"].lower() == "nan":
                continue
            points.append(
                {
                    "D": int(row["D"]),
                    "chi": int(row["chi"]),
                    "energy": float(row["energy"]),
                    "status": row.get("status", ""),
                    "source": row.get("source", ""),
                }
            )
    return sorted(points, key=lambda row: row["D"])


def site_xy(i: int, j: int) -> tuple[float, float]:
    """Honeycomb brickwall coordinates used by TeneT.jl's observable plot."""
    x = (j - 1) * math.sqrt(3.0) / 2.0
    y = -(i - 1) * 1.5 - (0.5 if (i + j) % 2 == 1 else 0.0)
    return x, y


def draw_bond_pattern(ax: plt.Axes, data: dict) -> None:
    bonds = data["d9_chi260_nearest_neighbor_bonds"]
    value_by_type_and_site: dict[tuple[str, int], float] = {}
    for bond in bonds:
        pval = int(PATTERN[bond["i"] - 1, bond["j"] - 1])
        value_by_type_and_site[(bond["bond_type"], pval)] = float(bond["energy"])

    n_i, n_j = 4, 12
    coords = {(i, j): site_xy(i, j) for i in range(1, n_i + 1) for j in range(1, n_j + 1)}

    def periodic_site_value(i: int, j: int) -> int:
        return int(PATTERN[(i - 1) % 2, (j - 1) % 6])

    drawn_values: list[float] = []
    for i in range(1, n_i + 1):
        for j in range(1, n_j + 1):
            pval = periodic_site_value(i, j)
            for bond_type, target in (
                ("bond_J1H_energy", (i, j + 1)),
                ("bond_J1V_energy", (i + 1, j)),
            ):
                if target not in coords or (bond_type, pval) not in value_by_type_and_site:
                    continue
                energy = value_by_type_and_site[(bond_type, pval)]
                drawn_values.append(energy)
                x1, y1 = coords[(i, j)]
                x2, y2 = coords[target]
                is_strong = energy < -0.30
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=STRONG if is_strong else WEAK,
                    linewidth=3.0 if is_strong else 1.2,
                    solid_capstyle="round",
                    zorder=1,
                )

    for x, y in coords.values():
        ax.scatter([x], [y], s=15, facecolor="white", edgecolor=TEXT, linewidth=0.55, zorder=3)

    original = [coords[(i, j)] for i in range(1, 3) for j in range(1, 7)]
    xs = [xy[0] for xy in original]
    ys = [xy[1] for xy in original]
    ax.add_patch(
        Rectangle(
            (min(xs) - 0.22, min(ys) - 0.24),
            max(xs) - min(xs) + 0.44,
            max(ys) - min(ys) + 0.48,
            fill=False,
            edgecolor="#6F7782",
            linewidth=0.8,
            linestyle=(0, (3.0, 2.2)),
            zorder=0,
        )
    )

    nine = sorted(float(bond["energy"]) for bond in bonds)
    strong_mean = float(np.mean(nine[:6]))
    weak_mean = float(np.mean(nine[6:]))
    contrast = weak_mean - strong_mean
    ax.text(
        0.02,
        0.04,
        rf"$\overline{{B}}_{{\rm strong}}={strong_mean:.5f}$"
        + "\n"
        + rf"$\overline{{B}}_{{\rm weak}}={weak_mean:.5f}$"
        + "\n"
        + rf"$\Delta_{{\rm PVB}}={contrast:.5f}$",
        transform=ax.transAxes,
        fontsize=6.6,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#D3D7DC", "linewidth": 0.5, "pad": 2.2},
        zorder=5,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=STRONG, lw=3.0, label="six strong $J_1$ bonds"),
            Line2D([0], [0], color=WEAK, lw=1.2, label="three weak $J_1$ bonds"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.94),
        frameon=False,
        fontsize=6.3,
        ncol=2,
        handlelength=2.2,
        columnspacing=1.2,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.40, max(x for x, _ in coords.values()) + 0.40)
    ax.set_ylim(min(y for _, y in coords.values()) - 1.55, 1.95)
    ax.axis("off")
    ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom")
    ax.set_title(r"brickwall cell: $D=9$", fontsize=7.5, pad=6.0)


def merge_site_xy(i: int, j: int, k: int) -> tuple[float, float]:
    """Honeycomb :merge coordinates, ported from TeneT.jl's plot_obs.jl
    (_honeycomb_merge_site_xy). A merge site (i, j) carries TWO physical spins,
    k=1 and k=2; the k=2 partner sits half a bond down-right of k=1."""
    x = (j - 1) * math.sqrt(3.0) + (i - 1) * math.sqrt(3.0) / 2.0
    y = -(i - 1) * 1.5
    if k == 1:
        return x, y
    return x + math.sqrt(3.0) / 2.0, y - 0.5


def circle_pair(ax: plt.Axes, a, b, color: str) -> None:
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    ax.add_patch(Ellipse((a + b) / 2, width=1.50, height=0.73,
                         angle=np.degrees(np.arctan2(delta[1], delta[0])),
                         fill=False, edgecolor=color, linewidth=0.95, zorder=4))


# Which spins a given J1 bond type connects, from _bond_offsets_honeycomb_merge:
# (k1, k2, offset of site 1, offset of site 2) in (di, dj).
MERGE_BONDS = {
    "bond_J1_onsite_energy": (1, 2, (0, 0), (0, 0)),
    "bond_J1H_energy":       (2, 1, (0, 0), (0, 1)),
    "bond_J1V_energy":       (2, 1, (0, 0), (1, 0)),
}


def draw_merge_bond_pattern(ax: plt.Axes, data: dict) -> None:
    blob = data["merge_pvb_d5_chi448_nearest_neighbor_bonds"]
    pattern = np.array(blob["cell"]["pattern_1based"], dtype=int)
    ni, nj = pattern.shape
    # value_by[(bond_type, pattern_value)] -> energy
    value_by: dict[tuple[str, int], float] = {}
    for bond_type, entries in blob["bonds"].items():
        for pos, energy in entries.items():
            oi, oj = (int(v) for v in pos.split(","))
            value_by[(bond_type, int(pattern[oi - 1, oj - 1]))] = float(energy)

    n_i, n_j = 4, 5
    coords = {(i, j, k): merge_site_xy(i, j, k)
              for i in range(1, n_i + 1) for j in range(1, n_j + 1) for k in (1, 2)}

    for i in range(1, n_i + 1):
        for j in range(1, n_j + 1):
            pval = int(pattern[(i - 1) % ni, (j - 1) % nj])
            for bond_type, (k1, k2, (d1i, d1j), (d2i, d2j)) in MERGE_BONDS.items():
                a = (i + d1i, j + d1j, k1)
                b = (i + d2i, j + d2j, k2)
                if a not in coords or b not in coords:
                    continue
                energy = value_by[(bond_type, pval)]
                x1, y1 = coords[a]
                x2, y2 = coords[b]
                is_strong = energy < -0.30
                ax.plot([x1, x2], [y1, y2],
                        color=STRONG if is_strong else WEAK,
                        linewidth=3.0 if is_strong else 1.2,
                        solid_capstyle="round", zorder=1)

    for (i, j, _k), (x, y) in coords.items():
        ax.scatter([x], [y], s=15, facecolor="white", edgecolor=TEXT,
                   linewidth=0.55, zorder=3)

    # lattice vectors: a1 along j, a2 along i (the cell is a parallelogram, not a box)
    a1 = (math.sqrt(3.0), 0.0)
    a2 = (math.sqrt(3.0) / 2.0, -1.5)
    ox, oy = merge_site_xy(1, 1, 1)
    ox, oy = ox - 0.30, oy + 0.30
    corners = [
        (ox, oy),
        (ox + nj * a1[0], oy + nj * a1[1]),
        (ox + nj * a1[0] + ni * a2[0], oy + nj * a1[1] + ni * a2[1]),
        (ox + ni * a2[0], oy + ni * a2[1]),
    ]
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="#6F7782",
                         linewidth=0.8, linestyle=(0, (3.0, 2.2)), zorder=0))
    for i in range(1, ni + 1):
        for j in range(1, nj + 1):
            circle_pair(ax, coords[(i, j, 1)], coords[(i, j, 2)],
                        PAIR_COLORS[pattern[i - 1, j - 1] - 1])
    label = (
        r"$\overline{B}_{\rm strong}=" + f"{blob['B_strong_mean']:.5f}" + "$\n"
        + r"$\overline{B}_{\rm weak}=" + f"{blob['B_weak_mean']:.5f}" + "$\n"
        + r"$\Delta_{\rm PVB}=" + f"{blob['Delta_PVB']:.5f}" + "$"
    )
    ax.text(
        0.01,
        0.02,
        label,
        transform=ax.transAxes,
        fontsize=6.6,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#D3D7DC", "linewidth": 0.5, "pad": 2.2},
        zorder=5,
    )
    ax.set_aspect("equal")
    allx = [x for x, _ in coords.values()]
    ally = [y for _, y in coords.values()]
    ax.set_xlim(min(allx) - 0.55, max(allx) + 0.35)
    ax.set_ylim(min(ally) - 1.55, max(ally) + 0.55)
    ax.axis("off")
    ax.text(0.0, 1.02, "(b)", transform=ax.transAxes, fontsize=8.0,
            fontweight="bold", va="bottom")
    ax.set_title(r"merge cell: $D=5$", fontsize=7.5, pad=6.0)


def draw_kagome_dimer_pattern(ax: plt.Axes, data: dict) -> None:
    """Kekule covering: three alternating dimers per disjoint strong hexagon.

    Contracting these pairs gives kagome vertices. The physical six-spin
    primitive cell is shown here; the one-hole 2x2 embedding adds a delta site.
    Bond widths encode measured groups, not individual orientation assignments.
    """
    t1 = np.array([3.0, 0.0])
    t2 = np.array([1.5, -3 * math.sqrt(3) / 2])
    angles = np.radians(np.arange(60, 420, 60))
    hexagon = np.column_stack((np.cos(angles), np.sin(angles)))
    sites = []
    hexagons = {}
    for i in range(2):
        for j in range(3):
            vertices = j * t1 + i * t2 + hexagon
            hexagons[(i, j)] = vertices
            sites.extend(vertices)
            for k in range(6):
                a, b = vertices[k], vertices[(k + 1) % 6]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=STRONG,
                        lw=3.0, solid_capstyle="round", zorder=1)
    sites = np.array(sites)
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            if a // 6 != b // 6 and np.isclose(np.linalg.norm(sites[a] - sites[b]), 1):
                ax.plot(sites[[a, b], 0], sites[[a, b], 1], color=WEAK,
                        lw=1.2, zorder=1)
    ax.scatter(sites[:, 0], sites[:, 1], s=15, facecolor="white",
               edgecolor=TEXT, linewidth=0.55, zorder=3)
    # Show the three inequivalent merged tensors and their periodic repeats.
    for vertices in hexagons.values():
        for role, k in enumerate((0, 2, 4)):
            circle_pair(ax, vertices[k], vertices[k + 1], PAIR_COLORS[role])
    # Wigner-Seitz primitive cell contains one entire six-spin plaquette.
    cell_angles = np.radians(np.arange(30, 390, 60))
    cell = math.sqrt(3) * np.column_stack((np.cos(cell_angles), np.sin(cell_angles)))
    ax.add_patch(Polygon(cell, fill=False, edgecolor="#6F7782", lw=0.8,
                         linestyle=(0, (3, 2.2)), zorder=0))
    bonds = data["kagome_dimer_d7_chi512_nearest_neighbor_bonds"]
    strong = np.mean([b["energy"] for b in bonds if b["group"] != "down_triangle"])
    weak = np.mean([b["energy"] for b in bonds if b["group"] == "down_triangle"])
    ax.text(0.01, 0.02,
            rf"$\overline{{B}}_{{\rm strong}}={strong:.5f}$" + "\n"
            + rf"$\overline{{B}}_{{\rm weak}}={weak:.5f}$" + "\n"
            + rf"$\Delta_{{\rm PVB}}={weak - strong:.5f}$",
            transform=ax.transAxes, fontsize=6.6, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#D3D7DC", "linewidth": 0.5, "pad": 2.2},
            zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, sites[:, 0].max() + 0.55)
    ax.set_ylim(sites[:, 1].min() - 1.85, 1.85)
    ax.axis("off")
    ax.text(-0.08, 1.02, "(c)", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom")
    ax.set_title(r"kagome-dimer cell: $D=7$", fontsize=7.5, pad=6)


def draw_energy_comparison(ax: plt.Axes, data: dict, merge: list[dict]) -> None:
    # Brickwall Plaquette branch, distinguished by its narrow diamond markers.
    brick = sorted(data["plaquette_best_by_D"], key=lambda row: row["D"])
    bx = np.array([row["D"] for row in brick], dtype=float)
    by = np.array([row["energy"] for row in brick])
    # Plaquette branch recomputed on the merge ansatz -- the like-for-like competitor.
    # The merge-cell points retain their existing display; their detailed chi
    # qualification is recorded in the JSON provenance.
    pvb = sorted(data["merge_pvb_best_by_D"], key=lambda row: row["D"])
    px = np.array([r["D"] for r in pvb], dtype=float)
    py = np.array([r["energy"] for r in pvb])
    mx = np.array([row["D"] for row in merge], dtype=float)
    my = np.array([row["energy"] for row in merge])

    ax.plot(bx, by, color=PLAQUETTE, linewidth=0.9, alpha=0.85,
            linestyle="-", zorder=1)
    ax.scatter(bx, by, marker="d", s=15, color=PLAQUETTE, edgecolor="white",
               linewidth=0.35, alpha=1.0, label="Plaquette, brickwall cell", zorder=2)

    ax.plot(px, py, color=PLAQUETTE, linewidth=0.9, alpha=0.85, zorder=3)
    ax.scatter(px, py, marker="D", s=26, color=PLAQUETTE,
               edgecolor="white", linewidth=0.45, label="Plaquette, merge cell", zorder=4)
    ax.plot(mx, my, color=MERGE, linewidth=0.9, alpha=0.8, zorder=1)
    ax.scatter(mx, my, marker="*", s=50, color=MERGE, edgecolor="white", linewidth=0.4,
               label="uniform merge", zorder=5)

    # Third ansatz for the SAME Plaquette state: the Kekule dimer covering merged
    # pair-by-pair onto a kagome lattice, contracted with the Setup-1 one-hole
    # embedding. Three inequivalent tensors and 3*D^4*4 parameters, i.e. the same
    # count as the merge-cell Plaquette points at equal D. Its bond pattern is the
    # same six-strong/three-weak order (Delta_PVB = 0.2253 against 0.2289 for the
    # D=9 brickwall reference), so this is the Plaquette branch, not a new state.
    kd = sorted(data["kagome_dimer_onehole_best_by_D"], key=lambda row: row["D"])
    kx = np.array([row["D"] for row in kd], dtype=float)
    ky = np.array([row["energy"] for row in kd])
    ax.plot(kx, ky, color=KDIMER, linewidth=0.85, alpha=0.75,
            linestyle="-", zorder=5)
    ax.scatter(kx, ky, marker="^", s=44, color=KDIMER, edgecolor="white",
               linewidth=0.45, label="Plaquette, kagome-dimer cell", zorder=7)

    ax.set_xlabel(r"bond dimension $D$")
    ax.set_ylabel(r"energy per site")
    ax.set_xlim(3.7, 10.35)
    ax.set_ylim(-0.44255, -0.43872)
    ax.set_xticks([4, 5, 6, 7, 8, 9, 10])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(alpha=0.22, linewidth=0.45)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "uniform merge")
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", frameon=False, fontsize=5.0, handletextpad=0.4,
              borderaxespad=0.15, labelspacing=0.28)
    ax.text(-0.13, 1.02, "(d)", transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom")
    ax.set_title(r"Variational energies at $J'_2/J_1=0.5$", fontsize=7.5, pad=8.0)


def main() -> None:
    data = load_data()
    merge = load_merge_points(data) + data["uniform_merge_like_for_like_best_by_D"]
    # Keep one point per D. For D=10 the user requested the latest 1x1 result,
    # defined here as the largest available environment chi (chi=1024).
    latest_by_D: dict[int, dict] = {}
    for row in merge:
        current = latest_by_D.get(row["D"])
        if current is None or row["chi"] > current["chi"]:
            latest_by_D[row["D"]] = row
    merge = [latest_by_D[D] for D in sorted(latest_by_D)]
    if [row["D"] for row in merge] != [4, 5, 6, 7, 8, 10]:
        raise RuntimeError(f"Expected uniform merge D=4,5,6,7,8,10 points, got {merge}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Three ansatz diagrams on the left; energies span all three rows.
    fig = plt.figure(figsize=(7.10, 6.10))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.18, 0.92])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[2, 0])
    ax_d = fig.add_subplot(grid[:, 1])
    draw_bond_pattern(ax_a, data)
    draw_merge_bond_pattern(ax_b, data)
    draw_kagome_dimer_pattern(ax_c, data)
    draw_energy_comparison(ax_d, data, merge)
    fig.subplots_adjust(left=0.030, right=0.985, bottom=0.085, top=0.93,
                        wspace=0.20, hspace=0.30)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
