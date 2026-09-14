from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parent
SOURCE_H5 = Path(
    r"D:\1 - research\1.24 - Honeycomb_J1J2\ED\results"
    r"\phase_kscan_summary_full.h5"
)
SUMMARY_CSV = HERE / "ed_phase_diagram_summary.csv"
SECTORS_CSV = HERE / "ed_phase_diagram_sectors.csv"
PROVENANCE_JSON = HERE / "ed_phase_diagram_provenance.json"
OUT_STEM = HERE / "ed_phase_diagram"

N_SITES = 24
EXPECTED_J2P = np.array(
    [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.66,
        0.67,
        0.68,
        0.69,
        0.70,
        0.71,
        0.72,
        0.73,
        0.74,
        0.75,
        0.76,
        0.77,
        0.78,
        0.79,
        0.80,
        0.90,
        1.00,
    ]
)

S12_MOMENTA = frozenset(
    {
        (0, 0),
        (1, 5),
        (2, 4),
        (3, 3),
        (4, 2),
        (5, 1),
        (3, 0),
        (4, 5),
        (5, 4),
        (0, 3),
        (1, 2),
        (2, 1),
    }
)

MOMENTUM_STYLE = {
    (0, 0): {
        "color": "#3B6EA8",
        "marker": "o",
        "label": r"$(n,m)=(0,0)$",
    },
    (2, 4): {
        "color": "#59A14F",
        "marker": "s",
        "label": r"$(n,m)=(2,4)$",
    },
    (4, 2): {
        "color": "#E6862A",
        "marker": "D",
        "label": r"$(n,m)=(4,2)$",
    },
}

LINE_COLOR = "#60666B"
CURVATURE_COLOR = "#3B6EA8"
BOUNDARY_COLOR = "#B23A48"
GRID_COLOR = "#AEB4B8"


def s12_momentum(qa: int, qb: int) -> tuple[int, int]:
    """Convert the source Z2 x Z6 character label to Fig. S12's (n,m).

    In the source lattice coordinates, the two primitive-cell translations
    used by Fig. S12 are represented by (1,1) and (0,-1). Matching their
    character phases gives n = 3*qa + qb and m = -qb, both modulo 6.
    """
    return (3 * int(qa) + int(qb)) % 6, (-int(qb)) % 6


def second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.full_like(y, np.nan, dtype=float)
    for index in range(1, len(x) - 1):
        h_left = x[index] - x[index - 1]
        h_right = x[index + 1] - x[index]
        result[index] = (
            2.0
            * (
                (y[index + 1] - y[index]) / h_right
                - (y[index] - y[index - 1]) / h_left
            )
            / (h_left + h_right)
        )
    return result


def orient_matrix(
    values: np.ndarray, expected_rows: int, expected_columns: int, name: str
) -> np.ndarray:
    if values.shape == (expected_rows, expected_columns):
        return values
    if values.shape == (expected_columns, expected_rows):
        return values.T
    raise ValueError(
        f"{name} has shape {values.shape}, expected "
        f"{(expected_rows, expected_columns)} or its transpose"
    )


def format_float(value: float) -> str:
    return "" if not np.isfinite(value) else repr(float(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refresh_dataset() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Refreshing the CSV files requires h5py; ordinary plotting does not."
        ) from exc

    if not SOURCE_H5.exists():
        raise FileNotFoundError(SOURCE_H5)

    with h5py.File(SOURCE_H5, "r") as source:
        j2p = np.asarray(source["J2p"], dtype=float).reshape(-1)
        raw_kpts = np.asarray(source["kpts"], dtype=int)
        if raw_kpts.shape[1] == 2:
            kpts = raw_kpts
        elif raw_kpts.shape[0] == 2:
            kpts = raw_kpts.T
        else:
            raise ValueError(f"kpts has unexpected shape {raw_kpts.shape}")

        n_j2p = len(j2p)
        n_sectors = len(kpts)
        energy_table = orient_matrix(
            np.asarray(source["E_table"], dtype=float),
            n_j2p,
            n_sectors,
            "E_table",
        )
        gs_index_julia = np.asarray(source["gs_kidx"], dtype=int).reshape(-1)
        gs_energy = np.asarray(source["gs_E"], dtype=float).reshape(-1)
        first_excited = np.asarray(source["e1_E"], dtype=float).reshape(-1)
        many_body_gap = np.asarray(source["gap"], dtype=float).reshape(-1)
        sector_dimensions = np.asarray(source["dims_k"], dtype=int).reshape(-1)
        metadata = {
            key: np.asarray(source["metadata"][key]).item()
            for key in source["metadata"].keys()
        }

    if np.min(gs_index_julia) < 1 or np.max(gs_index_julia) > n_sectors:
        raise ValueError("gs_kidx is not a valid Julia one-based sector index")
    gs_index = gs_index_julia - 1
    gamma_candidates = np.flatnonzero(np.all(kpts == np.array([0, 0]), axis=1))
    if len(gamma_candidates) != 1:
        raise ValueError("expected exactly one Gamma sector")
    gamma_index = int(gamma_candidates[0])

    energy_per_site = gs_energy / N_SITES
    curvature = second_derivative(j2p, energy_per_site)
    gamma_energy = energy_table[:, gamma_index]
    gamma_excess_per_site = (gamma_energy - gs_energy) / N_SITES

    summary_rows: list[dict[str, object]] = []
    for index, coupling in enumerate(j2p):
        qa, qb = (int(value) for value in kpts[gs_index[index]])
        n, m = s12_momentum(qa, qb)
        summary_rows.append(
            {
                "j2p_over_j1": float(coupling),
                "gs_sector_index": int(gs_index[index]),
                "gs_n": n,
                "gs_m": m,
                "source_gs_qa": qa,
                "source_gs_qb": qb,
                "gs_energy_total": float(gs_energy[index]),
                "gs_energy_per_site": float(energy_per_site[index]),
                "first_excited_energy_total": float(first_excited[index]),
                "many_body_gap": float(many_body_gap[index]),
                "gamma_energy_total": float(gamma_energy[index]),
                "gamma_excess_per_site": float(gamma_excess_per_site[index]),
                "energy_curvature": float(curvature[index]),
            }
        )

    sector_rows: list[dict[str, object]] = []
    for coupling_index, coupling in enumerate(j2p):
        for sector_index, (qa, qb) in enumerate(kpts):
            n, m = s12_momentum(int(qa), int(qb))
            sector_rows.append(
                {
                    "j2p_over_j1": float(coupling),
                    "sector_index": int(sector_index),
                    "n": n,
                    "m": m,
                    "source_qa": int(qa),
                    "source_qb": int(qb),
                    "sector_dimension": int(sector_dimensions[sector_index]),
                    "sector_energy_total": float(
                        energy_table[coupling_index, sector_index]
                    ),
                    "delta_energy_to_gs": float(
                        energy_table[coupling_index, sector_index]
                        - gs_energy[coupling_index]
                    ),
                }
            )

    validate(summary_rows, sector_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(SECTORS_CSV, sector_rows)

    provenance = {
        "source": {
            "path": str(SOURCE_H5),
            "sha256": sha256(SOURCE_H5),
            "original_plot_script": str(
                SOURCE_H5.parents[2] / "analysis" / "plot_ED_phase_diagram.jl"
            ),
        },
        "calculation": {
            "model": "spin-1/2 J1-J2p honeycomb Heisenberg model",
            "cluster": "24-site (2,2) C3-symmetric honeycomb torus",
            "J1": float(metadata["J1"]),
            "spin": float(metadata["S"]),
            "n_sites": int(metadata["N"]),
            "momentum_sectors": int(metadata["NK"]),
            "couplings": int(metadata["NJ"]),
        },
        "local_data": {
            "summary_csv": SUMMARY_CSV.name,
            "sector_csv": SECTORS_CSV.name,
            "summary_rows": len(summary_rows),
            "sector_rows": len(sector_rows),
        },
        "conventions": {
            "momentum": "(n,m) follow Fig. S12: k=(n*b1+m*b2)/6 modulo reciprocal lattice vectors",
            "source_momentum": "(qa,qb) are the internal Z2 x Z6 character labels stored in the source HDF5 file",
            "momentum_mapping": "(n,m)=((3*qa+qb) mod 6, (-qb) mod 6)",
            "gs_sector_index": "zero-based in the local CSV",
            "many_body_gap": "E1-E0 across all momentum sectors, total energy",
            "gamma_excess_per_site": "(E_Gamma-E0)/N",
            "delta_energy_to_gs": "E_min(n,m)-E0, total energy",
            "energy_curvature": "second derivative of E0/N on the nonuniform J2p/J1 grid",
        },
    }
    PROVENANCE_JSON.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return summary_rows, sector_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty dataset to {path}")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def load_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUMMARY_CSV.open(newline="", encoding="ascii") as handle:
        for raw in csv.DictReader(handle):
            source_qa = int(raw.get("source_gs_qa", raw.get("gs_qa", 0)))
            source_qb = int(raw.get("source_gs_qb", raw.get("gs_qb", 0)))
            mapped_n, mapped_m = s12_momentum(source_qa, source_qb)
            rows.append(
                {
                    "j2p_over_j1": float(raw["j2p_over_j1"]),
                    "gs_sector_index": int(raw["gs_sector_index"]),
                    "gs_n": int(raw.get("gs_n", mapped_n)),
                    "gs_m": int(raw.get("gs_m", mapped_m)),
                    "source_gs_qa": source_qa,
                    "source_gs_qb": source_qb,
                    "gs_energy_total": float(raw["gs_energy_total"]),
                    "gs_energy_per_site": float(raw["gs_energy_per_site"]),
                    "first_excited_energy_total": float(
                        raw["first_excited_energy_total"]
                    ),
                    "many_body_gap": float(raw["many_body_gap"]),
                    "gamma_energy_total": float(raw["gamma_energy_total"]),
                    "gamma_excess_per_site": float(raw["gamma_excess_per_site"]),
                    "energy_curvature": (
                        float(raw["energy_curvature"])
                        if raw["energy_curvature"]
                        else math.nan
                    ),
                }
            )
    return rows


def load_sectors() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SECTORS_CSV.open(newline="", encoding="ascii") as handle:
        for raw in csv.DictReader(handle):
            source_qa = int(raw.get("source_qa", raw.get("qa", 0)))
            source_qb = int(raw.get("source_qb", raw.get("qb", 0)))
            mapped_n, mapped_m = s12_momentum(source_qa, source_qb)
            rows.append(
                {
                    "j2p_over_j1": float(raw["j2p_over_j1"]),
                    "sector_index": int(raw["sector_index"]),
                    "n": int(raw.get("n", mapped_n)),
                    "m": int(raw.get("m", mapped_m)),
                    "source_qa": source_qa,
                    "source_qb": source_qb,
                    "sector_dimension": int(raw["sector_dimension"]),
                    "sector_energy_total": float(raw["sector_energy_total"]),
                    "delta_energy_to_gs": float(raw["delta_energy_to_gs"]),
                }
            )
    return rows


def validate(
    summary_rows: list[dict[str, object]],
    sector_rows: list[dict[str, object]],
) -> None:
    if len(summary_rows) != len(EXPECTED_J2P):
        raise ValueError(f"expected 25 summary rows, found {len(summary_rows)}")
    couplings = np.array([row["j2p_over_j1"] for row in summary_rows], dtype=float)
    if not np.allclose(couplings, EXPECTED_J2P, atol=1e-12):
        raise ValueError(f"unexpected coupling grid: {couplings}")
    if len(sector_rows) != len(EXPECTED_J2P) * 12:
        raise ValueError(f"expected 300 sector rows, found {len(sector_rows)}")

    sector_keys = [
        (row["j2p_over_j1"], row["sector_index"]) for row in sector_rows
    ]
    if len(sector_keys) != len(set(sector_keys)):
        raise ValueError("duplicate coupling-sector rows")

    mapped_momenta = {
        (int(row["n"]), int(row["m"])) for row in sector_rows
    }
    if mapped_momenta != S12_MOMENTA:
        raise ValueError(
            "the 12 source sectors do not map one-to-one onto the Fig. S12 "
            f"momentum set: {sorted(mapped_momenta)}"
        )

    for row in sector_rows:
        expected = s12_momentum(row["source_qa"], row["source_qb"])
        actual = (int(row["n"]), int(row["m"]))
        if actual != expected:
            raise ValueError(
                f"inconsistent Fig. S12 momentum mapping: {actual} != {expected}"
            )

    allowed_momenta = set(MOMENTUM_STYLE)
    gs_momenta = {
        (int(row["gs_n"]), int(row["gs_m"])) for row in summary_rows
    }
    if not gs_momenta <= allowed_momenta:
        raise ValueError(f"unexpected ground-state momenta: {gs_momenta}")

    by_key = {
        (float(row["j2p_over_j1"]), int(row["sector_index"])): row
        for row in sector_rows
    }
    for summary in summary_rows:
        key = (
            float(summary["j2p_over_j1"]),
            int(summary["gs_sector_index"]),
        )
        sector = by_key[key]
        if (int(summary["gs_n"]), int(summary["gs_m"])) != (
            int(sector["n"]),
            int(sector["m"]),
        ):
            raise ValueError(f"ground-state momentum mismatch at {key[0]}")
        if not math.isclose(
            float(sector["sector_energy_total"]),
            float(summary["gs_energy_total"]),
            abs_tol=1e-9,
        ):
            raise ValueError(f"ground-state energy mismatch at {key[0]}")


def finite_momentum_boundaries(
    summary_rows: list[dict[str, object]],
) -> tuple[float, float]:
    couplings = np.array([row["j2p_over_j1"] for row in summary_rows], dtype=float)
    is_gamma = np.array(
        [(row["gs_n"], row["gs_m"]) == (0, 0) for row in summary_rows]
    )
    switches = np.flatnonzero(is_gamma[1:] != is_gamma[:-1])
    if len(switches) != 2:
        raise ValueError(f"expected two Gamma/non-Gamma switches, found {switches}")
    return tuple(
        float((couplings[index] + couplings[index + 1]) / 2.0)
        for index in switches
    )


def style_axis(
    ax: plt.Axes,
    panel_label: str,
    boundaries: tuple[float, float],
) -> None:
    ax.set_box_aspect(0.76)
    ax.grid(color=GRID_COLOR, alpha=0.28, linewidth=0.45)
    ax.axvspan(
        boundaries[0],
        boundaries[1],
        color=BOUNDARY_COLOR,
        alpha=0.055,
        linewidth=0,
        zorder=0,
    )
    for boundary in boundaries:
        ax.axvline(
            boundary,
            color=BOUNDARY_COLOR,
            linestyle=(0, (3.0, 2.2)),
            linewidth=0.75,
            alpha=0.85,
            zorder=1,
        )
    ax.set_xlim(0.29, 1.01)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax.text(
        -0.12,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_momentum_colored(
    ax: plt.Axes,
    summary_rows: list[dict[str, object]],
    y_key: str,
    *,
    show_legend: bool = False,
    y_scale: float = 1.0,
) -> None:
    x_all = np.array([row["j2p_over_j1"] for row in summary_rows], dtype=float)
    y_all = y_scale * np.array([row[y_key] for row in summary_rows], dtype=float)
    ax.plot(x_all, y_all, color=LINE_COLOR, linewidth=0.9, alpha=0.72, zorder=2)

    for momentum, style in MOMENTUM_STYLE.items():
        selected = [
            row
            for row in summary_rows
            if (row["gs_n"], row["gs_m"]) == momentum
        ]
        if not selected:
            continue
        ax.scatter(
            [row["j2p_over_j1"] for row in selected],
            [y_scale * row[y_key] for row in selected],
            s=20,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.35,
            label=style["label"],
            zorder=4,
        )

    if show_legend:
        ax.legend(
            loc="upper left",
            frameon=False,
            handlelength=1.2,
            handletextpad=0.35,
            labelspacing=0.3,
            borderaxespad=0.2,
        )


def cell_edges(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def plot(
    summary_rows: list[dict[str, object]],
    sector_rows: list[dict[str, object]],
) -> None:
    validate(summary_rows, sector_rows)
    boundaries = finite_momentum_boundaries(summary_rows)

    fig = plt.figure(figsize=(7.0, 5.55), dpi=220)
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        height_ratios=(1.0, 1.0, 0.72),
        left=0.075,
        right=0.94,
        bottom=0.075,
        top=0.97,
        hspace=0.48,
        wspace=0.30,
    )
    ax_energy = fig.add_subplot(grid[0, 0])
    ax_curvature = fig.add_subplot(grid[0, 1])
    ax_gap = fig.add_subplot(grid[1, 0])
    ax_gamma = fig.add_subplot(grid[1, 1])
    ax_heatmap = fig.add_subplot(grid[2, 0:2])
    ax_colorbar = fig.add_subplot(grid[2, 2])

    for ax, label in zip(
        (ax_energy, ax_curvature, ax_gap, ax_gamma),
        ("(a)", "(b)", "(c)", "(d)"),
        strict=True,
    ):
        style_axis(ax, label, boundaries)

    plot_momentum_colored(
        ax_energy,
        summary_rows,
        "gs_energy_per_site",
        show_legend=True,
    )
    ax_energy.set_title("Ground-state energy", pad=2.5)
    ax_energy.set_ylabel(r"$E_0/N$")
    ax_energy.set_ylim(-0.483, -0.4325)

    finite_curvature = [
        row for row in summary_rows if np.isfinite(row["energy_curvature"])
    ]
    ax_curvature.plot(
        [row["j2p_over_j1"] for row in finite_curvature],
        [row["energy_curvature"] for row in finite_curvature],
        color=LINE_COLOR,
        linewidth=0.9,
        zorder=2,
    )
    ax_curvature.scatter(
        [row["j2p_over_j1"] for row in finite_curvature],
        [row["energy_curvature"] for row in finite_curvature],
        s=17,
        color=CURVATURE_COLOR,
        edgecolor="white",
        linewidth=0.3,
        zorder=4,
    )
    ax_curvature.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    ax_curvature.set_title("Energy curvature", pad=2.5)
    ax_curvature.set_ylabel(
        r"$\mathrm{d}^2(E_0/N)/\mathrm{d}(J'_2/J_1)^2$"
    )
    ax_curvature.set_ylim(-4.0, 0.2)

    plot_momentum_colored(ax_gap, summary_rows, "many_body_gap")
    ax_gap.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    ax_gap.set_title("Many-body gap", pad=2.5)
    ax_gap.set_xlabel(r"$J'_2/J_1$")
    ax_gap.set_ylabel(r"$E_1-E_0$")
    ax_gap.set_ylim(-0.025, 0.80)

    plot_momentum_colored(
        ax_gamma,
        summary_rows,
        "gamma_excess_per_site",
        y_scale=1.0e4,
    )
    ax_gamma.axhline(0.0, color="#777777", linewidth=0.55, zorder=1)
    ax_gamma.set_title(r"$\Gamma$-sector excess", pad=2.5)
    ax_gamma.set_xlabel(r"$J'_2/J_1$")
    ax_gamma.set_ylabel(r"$10^4(E_\Gamma-E_0)/N$")
    ax_gamma.set_ylim(-0.1, 3.1)

    couplings = np.array([row["j2p_over_j1"] for row in summary_rows], dtype=float)
    sectors = sorted(
        {
            (int(row["sector_index"]), int(row["n"]), int(row["m"]))
            for row in sector_rows
        }
    )
    values_by_key = {
        (float(row["j2p_over_j1"]), int(row["sector_index"])): float(
            row["delta_energy_to_gs"]
        )
        for row in sector_rows
    }
    matrix = np.array(
        [
            [values_by_key[(float(coupling), sector[0])] for sector in sectors]
            for coupling in couplings
        ]
    )
    order = np.argsort(matrix[0])
    ordered_sectors = [sectors[index] for index in order]
    ordered_matrix = matrix[:, order]
    x_edges = cell_edges(couplings)
    y_edges = np.arange(len(sectors) + 1, dtype=float) + 0.5

    heatmap = ax_heatmap.pcolormesh(
        x_edges,
        y_edges,
        ordered_matrix.T,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=0.2),
        shading="flat",
        rasterized=True,
    )
    for boundary in boundaries:
        ax_heatmap.axvline(
            boundary,
            color=BOUNDARY_COLOR,
            linestyle=(0, (3.0, 2.2)),
            linewidth=0.75,
            alpha=0.9,
        )
    ax_heatmap.set_xlim(0.29, 1.01)
    ax_heatmap.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax_heatmap.set_ylim(0.5, len(sectors) + 0.5)
    ax_heatmap.set_yticks(np.arange(1, len(sectors) + 1))
    ax_heatmap.set_yticklabels(
        [rf"$({n},{m})$" for _, n, m in ordered_sectors]
    )
    ax_heatmap.set_xlabel(r"$J'_2/J_1$")
    ax_heatmap.set_ylabel(r"$(n,m)$")
    ax_heatmap.set_title("Lowest energy in each momentum sector", pad=2.5)
    ax_heatmap.text(
        -0.055,
        1.05,
        "(e)",
        transform=ax_heatmap.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    colorbar = fig.colorbar(heatmap, cax=ax_colorbar)
    colorbar.set_label(r"$E_{(n,m)}-E_0$", labelpad=2)
    colorbar.set_ticks([0.0, 0.05, 0.10, 0.15, 0.20])

    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(OUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, **save_kwargs)
    plt.close(fig)

    print(
        "Finite-momentum ground-state window: "
        f"{boundaries[0]:.3f} < J2p/J1 < {boundaries[1]:.3f}"
    )
    print(OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="rebuild the local CSV files from the archived HDF5 scan",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    if args.refresh_data or not SUMMARY_CSV.exists() or not SECTORS_CSV.exists():
        summary_rows, sector_rows = refresh_dataset()
    else:
        summary_rows = load_summary()
        sector_rows = load_sectors()
    plot(summary_rows, sector_rows)


if __name__ == "__main__":
    main()
