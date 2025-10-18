# MASW Vs30 Estimation Project

## Project Overview
This project implements Multichannel Analysis of Surface Waves (MASW) to estimate 
near-surface shear-wave velocity (Vs) profiles and calculate Vs30 for seismic 
site classification.

## Directory Structure
```
masw_vs30_project/
├── data/
│   ├── raw/                    # Original data
│   │   ├── geophydog/         # Synthetic data from Geophydog
│   │   └── field/             # Field data (future)
│   ├── processed/             # Processed seismic data
│   └── dispersion_curves/     # Extracted dispersion curves
├── code/
│   ├── data_loading/          # Data I/O modules
│   ├── preprocessing/         # Signal processing
│   ├── dispersion_analysis/   # Dispersion extraction
│   ├── inversion/             # Inversion algorithms
│   ├── visualization/         # Plotting tools
│   └── utils/                 # Helper functions
├── notebooks/                 # Jupyter notebooks
├── results/
│   ├── figures/              # Output plots
│   ├── models/               # Vs profiles
│   └── reports/              # Analysis reports
└── docs/                     # Documentation
```

## Environment Setup
```bash
conda activate masw_env
```

## Dependencies
- Python 3.10
- NumPy, SciPy, Matplotlib, Pandas
- ObsPy (seismic data handling)
- disba (dispersion curve calculation)

## Data Source
Synthetic data from Geophydog: https://github.com/geophydog/Seismic_Data_Examples

## Project Phases
1. ✅ Background Research
2. ✅ Software Setup
3. ⏳ Data Exploration
4. ⏳ Dispersion Analysis
5. ⏳ Inversion
6. ⏳ Vs30 Calculation
7. ⏳ Final Report

## Author
[Your Name]

## License
[Your License Choice]
