# code/config.py
"""
Configuration file for MASW project
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "geophydog")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DISPERSION_DIR = os.path.join(DATA_DIR, "dispersion_curves")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Analysis parameters
DEFAULT_FREQ_MIN = 5.0   # Hz
DEFAULT_FREQ_MAX = 50.0  # Hz
DEFAULT_VEL_MIN = 100.0  # m/s
DEFAULT_VEL_MAX = 1000.0 # m/s

# Acquisition geometry (Geophydog data)
SOURCE_OFFSET = 10.0     # meters (x1)
RECEIVER_SPACING = 1.0   # meters (dx)
N_RECEIVERS = 60         # Number of geophones

# Distance correction factor
# SAC files have distances in km but stored as if they were m
# Actual: x1=10m, dx=1m but SAC headers show 0.01m, 0.001m
DISTANCE_CORRECTION_FACTOR = 1000.0  # Multiply SAC distances by this factor

# Vs30 calculation
VS30_DEPTH = 30.0        # meters

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")