# Cylinder VUMPS notebook

`data/VUMPS_data.ipynb` is the original notebook supplied with the manuscript, including embedded numerical arrays, Julia code, and saved outputs. Its recorded kernel is Julia 1.10.0. It is archived unchanged; it has not been executed or validated end to end in this data repository.

Cell numbers below are zero-based JSON cell indices, not execution counters.

| Cells | Content | Reproduction status |
| --- | --- | --- |
| 0–5 | Spin structure factor, embedded `cm05` matrix, helper functions | Output name is `SSF_D6400.pdf`; exact correspondence to the composite `Wrap-SSF.png` remains to be checked. Cell 3 uses `Ny` without a preceding top-level assignment. |
| 7–13 | Bond and chirality correlations, embedded arrays, connected and unconnected plots | Includes explicit save command for `MPS_correlations_unconnected.png`; not rerun. |
| 15–20 | Bond strengths and uniformity | Requires external `JLD2depot/HCInfZCdata_*.jld2` states; saves component plots rather than the composite `MPS-states.pdf`. |
| 22–26 | Entanglement and correlation-length scaling, embedded arrays | Explicit save commands for `EE_scaling.pdf` and `xi_scaling.pdf`; not rerun. |
| 28–36 | Strange correlators | Contains VUMPS optimization and loads `JLD2depot/U1HCInfZCdata_*.jld2`; cannot reproduce from the supplied files alone. |
| 39–44 | Flux-dependent correlation lengths and entropy scaling, embedded arrays | Saves `MPS_fluxedYC12_xi.png` and `MPS_fluxedYC12_S.png`; correspondence to composite `flux12.png` remains to be checked. |

The notebook imports MKL, JLD2, MAT, TOML, LinearAlgebra, MPSKit, Random, TensorKit, CairoMakie, Polynomials, KrylovKit, MPSKitModels, Printf, and LaTeXStrings. A Jupyter Julia kernel (IJulia) is needed for interactive use. No original Julia Project.toml/Manifest.toml accompanies this notebook, so package versions are not pinned. The Python requirements file does not install Julia dependencies.

Save commands target a relative `figs/` directory, which must exist under the notebook's working directory. Cells share state (for example, `flux_colors` is defined in cell 8 and reused later). Do not treat “Run All” as a lightweight plot-only operation: cells 29–30 initialize and optimize an MPS. Before independent reproduction, supply missing states and the original Julia environment, resolve initialization order, and verify generated panels against the manuscript.
