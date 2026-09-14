from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:\1 - research\1.24 - Honeycomb_J1J2")
DATA_CSV = HERE / "tm_spectrum_j2p_0p3_0p6.csv"
MINIMA_CSV = HERE / "tm_spectrum_j2p_0p3_0p6_minima.csv"
PROVENANCE_JSON = HERE / "tm_spectrum_j2p_0p3_0p6_provenance.json"
OUT_STEM = HERE / "tm_spectrum_j2p_0p3_0p6"
J2PS = (0.3, 0.4, 0.5, 0.6)

SOURCE_SPECS = (
    {
        "j2p": 0.3,
        "path": PROJECT_ROOT
        / "analysis/tm_spectrum_compare_20260714/"
        "tm_spectrum_J2p0p30_0p40_0p46_0p50_D8_chi256_kpi12.csv",
        "source_job": "43230054+43282536",
        "quality": "from-row",
        "quality_note": "Odd twelfths passed the 10/10 gate; older even-twelfth points are preliminary.",
    },
    {
        "j2p": 0.4,
        "path": PROJECT_ROOT
        / "hpc/rendered/2026-07-14_2105_bsc_J1J2p04_merge_D8chi256_TM_"
        "k0_pi12_pi_shared_env_tol1e6/pulled_remote/D8/TM_spectrum/trivial/"
        "spectrum_k0_pi12_pi.csv",
        "source_job": "43304744",
        "quality": "strict",
        "quality_note": "All 13 momenta passed the 10/10 eigensolver gate with one frozen converged environment.",
    },
    {
        "j2p": 0.5,
        "path": PROJECT_ROOT
        / "hpc/rendered/2026-07-16_1356_bsc_J1J2p05_merge_D8chi256_"
        "freshenv1e6_TM_orig_k0_pi12_pi/synced_final/D8/TM_spectrum/trivial/"
        "spectrum_k0_pi12_pi_orig_eig.csv",
        "source_job": "43368479",
        "quality": "original-eigsolve",
        "quality_note": "Fresh environment converged below 1e-6; original one-cycle Arnoldi semantics returned ten finite levels at every momentum.",
    },
    {
        "j2p": 0.6,
        "path": PROJECT_ROOT
        / "hpc/rendered/2026-07-18_102434_bsc_J1J2p06_merge_D8chi256_"
        "TM_JuliaEig_production/synced_final/D8/TM_spectrum/trivial/"
        "spectrum_k0_pi12_pi.csv",
        "source_job": "43470110",
        "quality": "original-eigsolve",
        "quality_note": "Production validation passed for 13 momenta and 130 finite levels using Julia one-cycle Arnoldi semantics.",
    },
)


def refresh_dataset() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        source_path = Path(spec["path"])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        with source_path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                raw_j2p = float(raw.get("J2p", spec["j2p"]))
                if not math.isclose(raw_j2p, float(spec["j2p"]), abs_tol=1e-10):
                    continue
                quality = raw.get("quality", "") if spec["quality"] == "from-row" else spec["quality"]
                source_job = (
                    raw.get("source_job", "")
                    if spec["quality"] == "from-row"
                    else spec["source_job"]
                )
                quality_note = (
                    raw.get("quality_note", "")
                    if spec["quality"] == "from-row"
                    else spec["quality_note"]
                )
                rows.append(
                    {
                        "j2p": float(spec["j2p"]),
                        "k_over_pi": float(raw["k_over_pi"]),
                        "k": float(raw["k"]),
                        "sector": raw["sector"],
                        "band": int(raw["band"]),
                        "gap": float(raw["gap"]),
                        "D": int(raw["D"]),
                        "chi_tm": int(raw["chi_tm"]),
                        "lattice": raw.get("lattice", "merge"),
                        "quality": quality,
                        "source_job": source_job,
                        "quality_note": quality_note,
                        "source_path": str(source_path),
                        "checkpoint": raw.get("checkpoint", ""),
                        "source_sha256": raw.get("source_sha256", ""),
                    }
                )

    validate(rows)
    columns = [
        "j2p",
        "k_over_pi",
        "k",
        "sector",
        "band",
        "gap",
        "D",
        "chi_tm",
        "lattice",
        "quality",
        "source_job",
        "quality_note",
        "source_path",
        "checkpoint",
        "source_sha256",
    ]
    with DATA_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in columns}
            for row in sorted(rows, key=lambda item: (item["j2p"], item["k_over_pi"], item["band"]))
        )

    provenance = {
        "configuration": {
            "model": "J1J2p Honeycomb merge iPEPS",
            "D": 8,
            "chi_tm": 256,
            "sector": "trivial",
            "bands": 10,
            "k_over_pi": [index / 12 for index in range(13)],
            "j2p": list(J2PS),
        },
        "sources": [
            {
                "j2p": spec["j2p"],
                "source_job": spec["source_job"],
                "quality": spec["quality"],
                "quality_note": spec["quality_note"],
                "path": str(spec["path"]),
            }
            for spec in SOURCE_SPECS
        ],
        "validation": {
            "rows": len(rows),
            "points_per_j2p": 13,
            "bands_per_point": 10,
            "duplicate_keys": 0,
        },
    }
    PROVENANCE_JSON.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def load_dataset() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "j2p": float(raw["j2p"]),
                    "k_over_pi": float(raw["k_over_pi"]),
                    "k": float(raw["k"]),
                    "band": int(raw["band"]),
                    "gap": float(raw["gap"]),
                    "D": int(raw["D"]),
                    "chi_tm": int(raw["chi_tm"]),
                }
            )
    validate(rows)
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    assert len(rows) == 520, f"expected 520 rows, got {len(rows)}"
    grouped: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["j2p"]), float(row["k_over_pi"]))].append(row)
    for j2p in J2PS:
        ks = sorted(k for (j, k) in grouped if math.isclose(j, j2p, abs_tol=1e-10))
        assert len(ks) == 13 and np.allclose(ks, np.arange(13) / 12), (j2p, ks)
        for k in ks:
            bands = sorted(int(row["band"]) for row in grouped[(j2p, k)])
            assert bands == list(range(1, 11)), (j2p, k, bands)
    keys = [(row["j2p"], row["k_over_pi"], row["band"]) for row in rows]
    assert len(keys) == len(set(keys)), "duplicate spectrum keys"
    assert all(np.isfinite(float(row["gap"])) for row in rows)


def branch(rows: list[dict[str, object]], j2p: float, band: int) -> list[dict[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if math.isclose(float(row["j2p"]), j2p, abs_tol=1e-10)
            and int(row["band"]) == band
        ),
        key=lambda row: float(row["k_over_pi"]),
    )


def write_minima(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    minima: list[dict[str, object]] = []
    for j2p in J2PS:
        point = min(branch(rows, j2p, 1), key=lambda row: float(row["gap"]))
        k_over_pi = float(point["k_over_pi"])
        if math.isclose(k_over_pi, 0.0):
            k_label = "Gamma"
        elif math.isclose(k_over_pi, 2.0 / 3.0):
            k_label = "K"
        else:
            k_label = f"{k_over_pi:.6g} pi"
        minima.append(
            {
                "j2p": j2p,
                "k_over_pi": k_over_pi,
                "k_label": k_label,
                "gap": float(point["gap"]),
                "quality": point["quality"],
                "source_job": point["source_job"],
            }
        )
    columns = ["j2p", "k_over_pi", "k_label", "gap", "quality", "source_job"]
    with MINIMA_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(minima)
    return minima


def plot(rows: list[dict[str, object]], minima: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.25), sharex=True, sharey=True, dpi=220)
    axes_flat = axes.ravel()
    low_color = "#3B6EA8"
    high_color = "#788690"
    min_color = "#B23A48"
    panel_labels = ("(a)", "(b)", "(c)", "(d)")

    for ax, j2p, panel_label, minimum in zip(
        axes_flat, J2PS, panel_labels, minima, strict=True
    ):
        ax.set_box_aspect(1)
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
                linewidth=1.25 if is_lowest else 0.55,
                alpha=1.0 if is_lowest else 0.42,
                zorder=2 if is_lowest else 1,
            )
            preliminary = np.array([point["quality"] == "preliminary" for point in points])
            if np.any(~preliminary):
                ax.scatter(
                    x[~preliminary],
                    y[~preliminary],
                    s=8 if is_lowest else 4.5,
                    color=color,
                    edgecolor="none",
                    alpha=1.0 if is_lowest else 0.46,
                    zorder=3,
                )
            if np.any(preliminary):
                ax.scatter(
                    x[preliminary],
                    y[preliminary],
                    s=9 if is_lowest else 5.5,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.45,
                    alpha=0.95 if is_lowest else 0.55,
                    zorder=3,
                )

        ax.axvline(2.0 / 3.0, color="#777777", linestyle=":", linewidth=0.6, alpha=0.65)
        ax.scatter(
            [minimum["k_over_pi"]],
            [minimum["gap"]],
            marker="*",
            s=34,
            color=min_color,
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )
        ax.set_title(rf"$J'_2/J_1={j2p:.1f}$", pad=2.5)
        ax.text(
            -0.16,
            1.03,
            panel_label,
            transform=ax.transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(0.25, 1.56)
        ax.set_xticks([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        ax.set_xticklabels(
            [
                r"$0(\Gamma)$",
                r"$\pi/3$",
                r"$2\pi/3(K)$",
                r"$\pi$",
            ]
        )
        ax.set_yticks([0.4, 0.8, 1.2, 1.6])
        ax.grid(alpha=0.22)

    axes[0, 0].set_ylabel(r"$\Delta_n(k)=-\ln|\lambda_n(k)|$")
    axes[1, 0].set_ylabel(r"$\Delta_n(k)=-\ln|\lambda_n(k)|$")
    axes[1, 0].set_xlabel(r"$k$")
    axes[1, 1].set_xlabel(r"$k$")
    axes[0, 0].text(
        0.04,
        0.95,
        "open: preliminary",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=4.6,
        color="#555555",
    )

    fig.tight_layout(h_pad=0.42, w_pad=0.45, pad=0.18)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(OUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="rebuild the local combined CSV from the archived calculation outputs",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 5.8,
            "axes.labelsize": 6.0,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = refresh_dataset() if args.refresh_data or not DATA_CSV.exists() else load_dataset()
    minima = write_minima(rows)
    plot(rows, minima)

    for minimum in minima:
        print(
            f"J2p={minimum['j2p']:.1f}: k/pi={minimum['k_over_pi']:.6g} "
            f"({minimum['k_label']}), gap={minimum['gap']:.9f}"
        )
    print(OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
