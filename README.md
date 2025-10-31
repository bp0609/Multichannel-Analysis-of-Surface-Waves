# Multichannel Analysis of Surface Waves (MASW)
## Vs30 Estimation and Seismic Site Classification

![Project Status](https://img.shields.io/badge/status-completed-success)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Data Source](#data-source)
- [Analysis Workflow](#analysis-workflow)
- [Results](#results)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [References](#references)
- [License](#license)

---

## 🎯 Project Overview

This project implements a complete **Multichannel Analysis of Surface Waves (MASW)** workflow to estimate near-surface shear-wave velocity (Vs) profiles and calculate **Vs30** for seismic site classification. The analysis pipeline processes seismic array data, extracts Rayleigh wave dispersion curves, inverts them to obtain layered earth models, and provides engineering interpretations for site classification according to international standards (NEHRP, Eurocode 8).

### Scientific Background

MASW is a non-invasive geophysical method that analyzes surface wave propagation to determine subsurface shear-wave velocity structure. The technique is widely used in:
- **Earthquake Engineering**: Site characterization for seismic design codes
- **Geotechnical Engineering**: Foundation design and soil profiling  
- **Environmental Studies**: Shallow subsurface investigation
- **Infrastructure Assessment**: Dam safety and infrastructure monitoring

The **Vs30** parameter (time-averaged shear-wave velocity in the top 30 meters) is a critical input for seismic hazard assessment and building code applications worldwide.

---

## ✨ Key Features

- ✅ **Complete MASW Processing Pipeline**: From raw seismic data to final site classification
- ✅ **Multiple Inversion Methods**: Least-squares, Monte Carlo global search, and hybrid approaches
- ✅ **Uncertainty Quantification**: Statistical analysis of model parameters and Vs30 estimates
- ✅ **International Standards**: Site classification per NEHRP, ASCE 7-22, and Eurocode 8
- ✅ **Comprehensive Visualization**: High-quality plots at every processing stage
- ✅ **Modular Code Structure**: Reusable components for data loading, processing, and analysis
- ✅ **Detailed Documentation**: Extensive logging and summary reports

---

## 📁 Project Structure

```
Multichannel-Analysis-of-Surface-Waves/
│
├── code/                              # Source code modules
│   ├── config.py                      # Project configuration and paths
│   │
│   ├── data_loading/                  # Data I/O modules
│   │   ├── explore_data.py            # Data exploration utilities
│   │   ├── load_sac.py                # SAC file loading functions
│   │   └── explore_data.ipynb         # Interactive data exploration
│   │
│   ├── preprocessing/                 # Signal processing
│   │   ├── signal_processing.py       # Filtering, normalization, whitening
│   │   └── run_preprocessing.py       # Main preprocessing script
│   │
│   ├── dispersion_analysis/           # Dispersion curve extraction
│   │   ├── phase_shift.py             # Phase shift (f-c) transform
│   │   └── extract_dispersion.py      # Dispersion extraction pipeline
│   │
│   ├── inversion/                     # Vs profile inversion
│   │   ├── forward_model.py           # Layered earth model & forward modeling
│   │   ├── initial_model.py           # Initial model generation
│   │   ├── least_square.py            # Damped least-squares inversion
│   │   ├── global_search.py           # Monte Carlo global search
│   │   ├── hybrid.py                  # Hybrid inversion approach
│   │   └── run_inversion.py           # Main inversion script
│   │
│   ├── vs30/                          # Vs30 calculation & classification
│   │   ├── calculate_vs30.py          # Vs30 and statistics calculation
│   │   ├── site_classification.py     # NEHRP & Eurocode 8 classification
│   │   ├── visualizations.py          # Vs30 visualization functions
│   │   └── run_vs30_analysis.py       # Main Vs30 analysis script
│   │
│   └── visualization/                 # General plotting utilities
│       └── vis.py                     # Visualization functions
│
├── data/                              # Data directory
│   ├── raw/                           # Original seismic data (60 SAC files)
│   ├── processed/                     # Preprocessed data (60 SAC files)
│   │   └── processing_log.txt         # Processing parameters log
│   └── dispersion_curves/             # Analysis results
│       ├── dispersion_curve_fundamental.txt
│       ├── dispersion_analysis_summary.txt
│       ├── vs_profile_final.txt
│       ├── inversion_summary.txt
│       └── site_characterization_report.txt
│
├── results/                           # Output products
│   └── figures/                       # All generated figures
│       ├── acquisition_geometry.png
│       ├── shot_gather_raw.png
│       ├── preprocessing_comparison.png
│       ├── dispersion_phase_shift.png
│       ├── dispersion_picked_auto.png
│       ├── inversion/
│       │   ├── observed_dispersion.png
│       │   ├── initial_model.png
│       │   ├── result_least_squares.png
│       │   ├── result_monte_carlo.png
│       │   ├── result_hybrid.png
│       │   ├── comparison_all.png
│       │   └── uncertainty_envelope.png
│       └── vs30/
│           ├── vs_profile_with_vs30.png
│           ├── vs_statistics.png
│           ├── nehrp_classification.png
│           ├── vs30_uncertainty.png
│           └── summary_report.png
│
├── inspect_geophydog_data.py          # Data inspection utility
├── test_disba.py                      # disba library testing
├── README.md                          # This file
└── LICENSE                            # Project license

```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10 or higher

### Environment Setup

**Clone the repository**:
```bash
git clone https://github.com/bp0609/Multichannel-Analysis-of-Surface-Waves.git
cd Multichannel-Analysis-of-Surface-Waves
```

### Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Numerical computing |
| scipy | ≥1.7 | Scientific computing, optimization |
| matplotlib | ≥3.4 | Plotting and visualization |
| pandas | ≥1.3 | Data handling |
| obspy | ≥1.3 | Seismic data I/O (SAC files) |
| disba | ≥0.5 | Surface wave dispersion modeling |

---

## 📊 Data Source

**Synthetic MASW Data from Geophydog**

- **Source**: [geophydog/Seismic_Data_Examples](https://github.com/geophydog/Seismic_Data_Examples)
- **Format**: SAC (Seismic Analysis Code) binary files
- **Array Configuration**:
  - Source offset (x1): 10.0 m
  - Number of receivers: 60
  - Receiver spacing (dx): 1.0 m
  - Total array length: 59.0 m
  - Acquisition type: Active source (shot gather)

- **Acquisition Parameters**:
  - Sampling rate: 512 Hz
  - Record length: 8 seconds
  - Data format: Vertical component seismograms

**Note**: SAC file distance headers have a unit scaling issue (stored in km instead of m). The code applies a correction factor of 1000× to all distances.

---

## 🔄 Analysis Workflow

### Phase 1️⃣: Data Loading & Exploration
**Script**: `code/data_loading/explore_data.py`

- Load 60-channel SAC files
- Extract acquisition geometry from SAC headers
- Visualize shot gather (time-distance plot)
- Analyze frequency content and signal quality

**Key Outputs**:
- `shot_gather_raw.png`: Raw seismic data display
- `acquisition_geometry.png`: Receiver array layout
- `frequency_analysis_comprehensive.png`: Spectral content

---

### Phase 2️⃣: Preprocessing
**Script**: `code/preprocessing/run_preprocessing.py`

**Processing Steps**:
1. **Bandpass Filtering**: 5-50 Hz (4th-order Butterworth, zero-phase)
2. **Trace Normalization**: Amplitude normalization per trace
3. **Quality Control**: Remove NaN/Inf values

**Parameters**:
```python
freqmin = 5.0   # Hz (low-frequency cutoff)
freqmax = 50.0  # Hz (high-frequency cutoff)
norm_method = 'trace'  # Normalize each trace independently
```

**Outputs**:
- 60 processed SAC files in `data/processed/`
- `preprocessing_comparison.png`: Before/after comparison
- `processing_log.txt`: Processing parameters

---

### Phase 3️⃣: Dispersion Analysis
**Script**: `code/dispersion_analysis/extract_dispersion.py`

**Method**: Phase Shift (f-c) Transform

The phase shift method (also called f-c or τ-p transform) analyzes the phase coherency of surface waves across the receiver array as a function of frequency and phase velocity.

**Analysis Parameters**:
```python
freq_min = 5.0 Hz      # Minimum frequency
freq_max = 50.0 Hz     # Maximum frequency
n_freqs = 450          # Frequency sampling points

vel_min = 100.0 m/s    # Minimum phase velocity
vel_max = 800.0 m/s    # Maximum phase velocity
n_vels = 500           # Velocity sampling points
```

**Dispersion Picking**: Automatic peak detection with uncertainty estimation

**Results**:
- **Velocity range**: 174.1 - 664.8 m/s
- **Frequency range**: 5.0 - 50.0 Hz
- **Mean uncertainty**: 16.5 m/s
- **Wavelength range**: 4.8 - 119.9 m
- **Estimated max depth**: ~120 m (λ_max / 2)

**Outputs**:
- `dispersion_curve_fundamental.txt`: Picked dispersion curve
- `dispersion_phase_shift.png`: Dispersion image
- `dispersion_picked_auto.png`: Picked curve overlay
- `dispersion_analysis_summary.txt`: Analysis report

---

### Phase 4️⃣: Inversion for Vs Profile
**Script**: `code/inversion/run_inversion.py`

**Forward Modeling**: Compute theoretical Rayleigh wave dispersion using `disba` (modal summation method)

**Inversion Approaches**:

#### 1. **Least-Squares Inversion**
- Method: Damped iterative least-squares (Levenberg-Marquardt)
- Objective: Minimize data misfit with regularization
- Result: RMS error = 72.72 m/s, Vs30 = 309.4 m/s

#### 2. **Monte Carlo Global Search**
- Method: Random sampling with acceptance criteria
- Models tested: 1000
- Acceptable models: 100 (RMS < threshold)
- Best result: RMS error = 37.46 m/s, Vs30 = 343.2 m/s

#### 3. **Hybrid Approach (FINAL)**
- Combine global search exploration with local optimization
- Result: **RMS error = 73.16 m/s, Vs30 = 311.6 m/s**

**Earth Model Parameterization**:
- Number of layers: 6 (5 layers + half-space)
- Free parameters: Vs, layer thickness
- Constrained parameters: Vp (from Vs using empirical relations), density (ρ)

**Outputs**:
- `vs_profile_final.txt`: Final layered earth model
- `inversion/result_hybrid.png`: Best-fit model and dispersion
- `inversion/comparison_all.png`: All three methods compared
- `inversion/uncertainty_envelope.png`: Ensemble uncertainty
- `inversion_summary.txt`: Inversion report

---

### Phase 5️⃣: Vs30 Calculation & Site Classification
**Script**: `code/vs30/run_vs30_analysis.py`

**Vs30 Calculation**:

$$\text{Vs30} = \frac{30}{\sum_{i=1}^{n} \frac{h_i}{V_{si}}}$$

where $h_i$ is layer thickness and $V_{si}$ is shear-wave velocity for layer $i$.

**Statistical Analysis**:
- Vs30 = **311.6 m/s** (final model)
- Vs30 mean = 322.9 m/s (from 100 Monte Carlo models)
- Vs30 std dev = 44.6 m/s
- 95% Confidence Interval = [235.4, 410.4] m/s
- Coefficient of Variation = 13.8%

**Other Metrics**:
- Vs10 = 287.3 m/s
- Vs15 = 287.3 m/s
- Vs20 = 287.3 m/s
- Surface Vs = 287.3 m/s
- Maximum Vs = 1145.3 m/s (half-space)

**Site Classifications**:

| Standard | Site Class | Description | Vs30 Range |
|----------|-----------|-------------|------------|
| **NEHRP (ASCE 7-22)** | **D** | **Stiff Soil** | 180-360 m/s |
| NEHRP Extended | D | Stiff Soil | 180-360 m/s |
| Eurocode 8 | C | Dense sand, gravel, or stiff clay | 180-360 m/s |

**Site Coefficients** (for design spectrum):
- Fa (short period amplification) = 1.60
- Fv (long period amplification) = 2.40

**Outputs**:
- `vs30/vs_profile_with_vs30.png`: Vs profile with Vs30 overlay
- `vs30/vs_statistics.png`: Multiple Vs metrics
- `vs30/nehrp_classification.png`: Classification chart
- `vs30/vs30_uncertainty.png`: Uncertainty analysis
- `vs30/summary_report.png`: Comprehensive summary
- `site_characterization_report.txt`: Full engineering report

---

## 📈 Results

### Final Shear-Wave Velocity Profile

| Layer | Thickness (m) | Vs (m/s) | Vp (m/s) | Density (g/cm³) |
|-------|---------------|----------|----------|-----------------|
| 1 | 25.43 | 287.3 | 497.1 | 0.26 |
| 2 | 3.87 | 564.1 | 975.8 | 0.31 |
| 3 | 4.57 | 768.5 | 1329.5 | 0.33 |
| 4 | 25.44 | 1053.6 | 1822.7 | 0.36 |
| 5 | 32.19 | 1055.7 | 1826.3 | 0.36 |
| 6 | ∞ (half-space) | 1145.3 | 1981.3 | 0.37 |

### Key Findings

✅ **Vs30 = 311.6 m/s** → **NEHRP Site Class D (Stiff Soil)**

✅ **Engineering Implications**:
- Moderate amplification of seismic ground motions expected
- Standard seismic design provisions typically adequate
- Liquefaction potential should be evaluated in saturated zones
- Conventional foundation systems usually suitable
- Site-specific response analysis recommended for critical facilities

✅ **Quality Metrics**:
- Maximum investigation depth: ~120 m
- Coverage for Vs30 calculation: **ADEQUATE** (well exceeds 30 m requirement)
- Inversion RMS error: 73.16 m/s (~11% relative error)
- Uncertainty analysis: 100 Monte Carlo models evaluated

---

## 🚀 How to Run

### Complete Workflow (All Phases)

```bash
# Navigate to project directory
cd /path/to/Multichannel-Analysis-of-Surface-Waves

# Phase 1: Data Exploration
python code/data_loading/explore_data.py

# Phase 2: Preprocessing
python code/preprocessing/run_preprocessing.py

# Phase 3: Dispersion Analysis
python code/dispersion_analysis/extract_dispersion.py

# Phase 4: Inversion
python code/inversion/run_inversion.py

# Phase 5: Vs30 Analysis
python code/vs30/run_vs30_analysis.py
```

---

## 📤 Outputs

### Data Products

1. **Dispersion Curves**:
   - `data/dispersion_curves/dispersion_curve_fundamental.txt`
   - Format: frequency (Hz), phase velocity (m/s), uncertainty (m/s)

2. **Vs Profile**:
   - `data/dispersion_curves/vs_profile_final.txt`
   - Format: layer number, thickness (m), Vs (m/s), Vp (m/s), density (g/cm³)

3. **Reports**:
   - `dispersion_analysis_summary.txt`: Dispersion extraction details
   - `inversion_summary.txt`: Inversion results and comparison
   - `site_characterization_report.txt`: Full engineering report with site classification

### Figures

All figures are saved in `results/figures/` with publication-quality resolution (300 DPI):

**Data Exploration**:
- `shot_gather_raw.png`
- `acquisition_geometry.png`
- `frequency_content.png`

**Preprocessing**:
- `preprocessing_comparison.png`

**Dispersion Analysis**:
- `dispersion_phase_shift.png`
- `dispersion_picked_auto.png`

**Inversion** (in `inversion/` subdirectory):
- `observed_dispersion.png`
- `initial_model.png`
- `result_least_squares.png`
- `result_monte_carlo.png`
- `result_hybrid.png`
- `comparison_all.png`
- `uncertainty_envelope.png`

**Vs30 Analysis** (in `vs30/` subdirectory):
- `vs_profile_with_vs30.png`
- `vs_statistics.png`
- `nehrp_classification.png`
- `vs30_uncertainty.png`
- `summary_report.png`

---

## 📚 References

### Standards & Guidelines

1. **ASCE 7-22**: Minimum Design Loads and Associated Criteria for Buildings and Other Structures
2. **NEHRP**: Recommended Seismic Provisions for New Buildings and Other Structures (FEMA P-2082)
3. **Eurocode 8**: Design of structures for earthquake resistance - Part 1: General rules, seismic actions and rules for buildings

### Scientific Literature

1. Park, C. B., Miller, R. D., & Xia, J. (1999). Multichannel analysis of surface waves. *Geophysics*, 64(3), 800-808.
2. Xia, J., Miller, R. D., & Park, C. B. (1999). Estimation of near-surface shear-wave velocity by inversion of Rayleigh waves. *Geophysics*, 64(3), 691-700.
3. Socco, L. V., & Boiero, D. (2008). Improved Monte Carlo inversion of surface wave data. *Geophysical Prospecting*, 56(3), 357-371.

### Software & Tools

- **ObsPy**: Seismology processing framework - [https://obspy.org/](https://obspy.org/)
- **disba**: Surface wave dispersion in layered media - [https://github.com/keurfonluu/disba](https://github.com/keurfonluu/disba)
- **Geophydog Data**: Example seismic datasets - [https://github.com/geophydog/Seismic_Data_Examples](https://github.com/geophydog/Seismic_Data_Examples)

---

## 👥 Contributors

- **Developer**: GeoPhy Course Project
- **Repository**: [bp0609/Multichannel-Analysis-of-Surface-Waves](https://github.com/bp0609/Multichannel-Analysis-of-Surface-Waves)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Synthetic data provided by [geophydog](https://github.com/geophydog)
- Surface wave dispersion modeling using [disba](https://github.com/keurfonluu/disba)
- Seismic data handling via [ObsPy](https://obspy.org/)

---

## 📧 Contact

For questions or collaborations, please open an issue on the [GitHub repository](https://github.com/bp0609/Multichannel-Analysis-of-Surface-Waves/issues).

---

