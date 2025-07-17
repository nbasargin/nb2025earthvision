# Explainable Physical PolSAR Autoencoders for Soil Moisture Estimation

This repository contains code to reproduce the results presented at the EarthVision 2025 CVPR Workshop.

The code is organized into a python package. To reproduce the results, access to the data from experimental
F-SAR `CROPEX 2014` and `HTERRA 2022` campaigns is required.

## Setup
- clone this repository
- create a python environment, e.g. `conda create -n nb2025earthvision python=3.13`
- install this package by running `pip install -e .` in the repository root folder (where `pyproject.toml` is located)
- helper `fsarcamp` and `sarssm` packages will be installed from github
- adjust paths to F-SAR data and ground measurements in `campaign_data.py`
- run `python -m nb2025earthvision` to reproduce all experiments and figures

## Details
Datasets are generated from the F-SAR campaign data and cached with `python -m nb2025earthvision.datasets`.

After the datasets are cached, the experiments can be run individually:
- Physical model calibration: `python -m nb2025earthvision.experiment_1_calibration`
- Physical model inversion: `python -m nb2025earthvision.experiment_2_physical`
- Supervised model training: `python -m nb2025earthvision.experiment_3_supervised`
- Self-supervised autoencoder training: `python -m nb2025earthvision.experiment_4_selfsupervised`
- Hybrid autoencoder training: `python -m nb2025earthvision.experiment_5_hybrid`

Paper figures summarize the experimental results and are created with `python -m nb2025earthvision.paper_figures`.

Additional code and functionality are organized as follows:
- `campaign_data.py`: data interface to the campaign data, mostly relies on the `fsarcamp` package
- `constants.py`: constants including calibrated parameters, feasible value ranges, etc.
- `metrics.py`: distance and accuracy metrics
- `models.py`: implementation of ML and physical models, relies on the `sarssm` package for moisture conversion
- `paths.py`: folder structure definition for output results
- `plot_functions.py`: plot and figure helpers
- `regions.py`: definition of geographical regions and fields, mostly relies on the `fsarcamp` package
- `validation.py`: performance validation helpers
