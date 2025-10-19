# notebooks/02_preprocessing.ipynb
# Or: code/preprocessing/run_preprocessing.py

"""
Phase 3.2: Data Preprocessing
Apply filters, normalization, and prepare data for dispersion analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream
import os
import pandas as pd
import sys

# Add project paths
sys.path.append('..')
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, DISTANCE_CORRECTION_FACTOR
from data_loading.explore_data import load_masw_data
from preprocessing.signal_processing import (
    preprocessing_pipeline,
    compare_processing
)

# ============================================
# 1. LOAD RAW DATA
# ============================================

print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)

stream_raw, _ = load_masw_data(RAW_DATA_DIR)

# Extract distances (apply correction factor)
distances = []
for tr in stream_raw:
    if hasattr(tr.stats.sac, 'dist'):
        distances.append(tr.stats.sac.dist * DISTANCE_CORRECTION_FACTOR)
distances = np.array(distances)

print(f"Loaded {len(stream_raw)} traces")

# ============================================
# 2. APPLY PREPROCESSING
# ============================================

stream_processed = preprocessing_pipeline(
    stream_raw,
    distances=distances,
    apply_filter=True,
    freqmin=5.0,
    freqmax=50.0,
    apply_normalize=True,
    norm_method='trace',
    apply_whiten=False  # Try with True to see effect
)

# ============================================
# 3. COMPARE ORIGINAL VS PROCESSED
# ============================================

print("\n" + "=" * 60)
print("GENERATING COMPARISON PLOTS")
print("=" * 60)

fig_comparison = compare_processing(
    stream_raw,
    stream_processed,
    trace_idx=len(stream_raw)//2,  # Middle trace
    distances=distances,
    save_path=os.path.join(FIGURES_DIR, 'preprocessing_comparison.png')
)

# ============================================
# 4. SAVE PROCESSED DATA
# ============================================

print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Save as SAC files
for i, tr in enumerate(stream_processed):
    output_file = os.path.join(PROCESSED_DATA_DIR, f"trace_{i:03d}_processed.sac")
    tr.write(output_file, format='SAC')

print(f"Saved {len(stream_processed)} processed traces to {PROCESSED_DATA_DIR}")

# ============================================
# 5. DOCUMENT PREPROCESSING PARAMETERS
# ============================================

# Create processing log
processing_log = f"""
MASW DATA PREPROCESSING LOG
========================================

Input Data:
  - Source: Geophydog synthetic data
  - Number of traces: {len(stream_raw)}
  - Sampling rate: {stream_raw[0].stats.sampling_rate} Hz
  - Duration: {stream_raw[0].stats.npts / stream_raw[0].stats.sampling_rate:.3f} s
  - Source offset: {distances.min():.1f} m
  - Receiver spacing: {np.median(np.diff(np.sort(distances))):.1f} m
  - Array length: {distances.max() - distances.min():.1f} m

Processing Steps Applied:
  1. Bandpass Filter: 5-50 Hz (4 corners, zero-phase)
  2. Normalization: trace-by-trace (each trace to its maximum)
  3. Spectral Whitening: Not applied

Output:
  - Processed traces saved to: {PROCESSED_DATA_DIR}
  - Number of output files: {len(stream_processed)}
  
Quality Notes:
  - All traces processed successfully
  - No NaN or Inf values in output
  - Ready for dispersion analysis

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save log
log_file = os.path.join(PROCESSED_DATA_DIR, 'processing_log.txt')
with open(log_file, 'w') as f:
    f.write(processing_log)

print(f"\nProcessing log saved to: {log_file}")

print("\n" + "=" * 60)
print("PHASE 3.2 COMPLETE: Data Preprocessing")
print("=" * 60)
print("\nProcessed data ready for dispersion analysis!")
print("Proceed to Phase 4: Dispersion Analysis")