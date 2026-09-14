# Data for HoneycombFlat

Supporting numerical data for **Extended symmetric regime in a honeycomb Heisenberg model with sublattice-selective interactions**, by Nai Chao Hu, Xing-Yu Zhang, Yuchi He, and Nick Bultinck.

This is a private, pre-arXiv preparation snapshot. It is not yet a complete archive of every figure. The repository will be made public after the arXiv preprint appears; the arXiv identifier and final data version must then be recorded here. No public release has been made.

## Contents and reproduction

`data/` contains the existing manuscript CSV/JSON tables, provenance records, plotting scripts, and the original Julia `VUMPS_data.ipynb` notebook, copied without changes. CSV headers define the stored fields; JSON records and script selection rules retain the model, bond dimensions, observable conventions, exclusions, and source information. Historical source paths identify provenance and are not required for ordinary Python plotting. Large simulation checkpoints and full run histories are not included.

The VUMPS notebook includes embedded cylinder data and saved outputs, but also requires external JLD2 states for some sections and contains optimization cells. It has not been rerun. See `NOTEBOOK_GUIDE.md` for cell-level coverage, Julia dependencies, and remaining reproduction gaps.

Install Python 3.10 or newer and the packages in `requirements.txt`. Run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python data/plot_fig2_j2p_magnetization.py
python data/plot_j2p0p5_invxi_tm_spectrum.py
python data/plot_plaquette_vbs_comparison.py
python data/plot_tm_spectrum_j2p.py
python data/plot_ed_phase_diagram.py
python data/plot_ipeps_merge_scheme.py
```

The scripts write PDF/PNG figures beside the data. Do not use `--refresh-data`: that option requires the original research directories. The standalone `plot_j2p0p5_invxi_vs_invD.py` produces an additional diagnostic and rewrites its fit summary. Numerical tables are the reproducibility reference; PDF metadata and rendering can vary by environment. The correlation-length extrapolation is sensitive to the selected bond dimensions; retain the selections and caveats in the scripts.

`FIGURE_COVERAGE.md` lists each active manuscript figure and the remaining missing sources. `MANIFEST.json` records SHA256 hashes of the supplied files, manuscript revision, and working-tree snapshot provenance.

## Citation and release

Repository: https://github.com/XingyuZhang2018/data_for_HoneycombFlat

Citation metadata is in `CITATION.cff`. There is no DOI or arXiv identifier assigned in this snapshot. Before public release, complete the missing sources, verify correspondence with the arXiv figures, agree a data/code license with the authors, and tag the final snapshot. No license is assigned during private preparation.
