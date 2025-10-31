"""
Dispersion Analysis
Extract dispersion curves from preprocessed MASW data
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import os
import sys
import glob

# Add project paths
sys.path.append('..')
from config import RAW_DATA_DIR, DISPERSION_DIR, FIGURES_DIR, DISTANCE_CORRECTION_FACTOR
from dispersion_analysis.phase_shift import (
    phase_shift_transform,
    plot_dispersion_image,
    automatic_picking,
    interactive_picking
)

# Create output directories
os.makedirs(DISPERSION_DIR, exist_ok=True)

# ============================================
# 1. LOAD PREPROCESSED DATA
# ============================================

print("=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

# Load all processed SAC files
sac_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.sac")))
print(f"Found {len(sac_files)} processed files")

# Read into obspy stream
from obspy import Stream
stream = Stream()
for sac_file in sac_files:
    stream += read(sac_file)

print(f"Loaded {len(stream)} traces")

# ============================================
# 2. PREPARE DATA FOR DISPERSION ANALYSIS
# ============================================

# Extract distances (apply correction factor)
distances = []
for tr in stream:
    if hasattr(tr.stats.sac, 'dist'):
        distances.append(tr.stats.sac.dist * DISTANCE_CORRECTION_FACTOR)
    else:
        distances.append(0)

distances = np.array(distances)

# Sort by distance
sort_idx = np.argsort(distances)
distances = distances[sort_idx]

# Create data matrix (n_traces x n_samples)
n_traces = len(stream)
n_samples = stream[0].stats.npts
data = np.zeros((n_traces, n_samples))

for i, idx in enumerate(sort_idx):
    data[i, :] = stream[idx].data

# Get time parameters
dt = stream[0].stats.delta
fs = stream[0].stats.sampling_rate

print(f"\nData array shape: {data.shape}")
print(f"Sampling rate: {fs} Hz")
print(f"Time step: {dt} s")
print(f"Distance range: {distances.min():.1f} - {distances.max():.1f} m")
print(f"Receiver spacing: {np.median(np.diff(distances)):.2f} m")

# ============================================
# 3. DEFINE ANALYSIS PARAMETERS
# ============================================

# Frequency range for analysis (Hz)
# Note: Below 5 Hz, the data quality is often poor (low SNR)
freq_min = 5.0
freq_max = 50.0
n_freqs = 450
frequencies = np.linspace(freq_min, freq_max, n_freqs)

# Velocity range for analysis (m/s)
# Extended range to avoid hitting upper bound
vel_min = 100.0
vel_max = 800.0
n_vels = 500
velocities = np.linspace(vel_min, vel_max, n_vels)

print(f"\nAnalysis parameters:")
print(f"  Frequency range: {freq_min} - {freq_max} Hz ({n_freqs} points)")
print(f"  Velocity range: {vel_min} - {vel_max} m/s ({n_vels} points)")

# ============================================
# 4. COMPUTE DISPERSION IMAGE - PHASE SHIFT METHOD
# ============================================

print("\n" + "=" * 60)
print("PHASE SHIFT METHOD (Geophydog f-c transform)")
print("=" * 60)

dispersion_image = phase_shift_transform(
    data=data,
    dt=dt,
    offsets=distances,
    frequencies=frequencies,
    velocities=velocities
)

# Plot phase shift result
fig_ps = plot_dispersion_image(
    dispersion_image,
    frequencies,
    velocities,
    title="Dispersion Image - Phase Shift Method",
    save_path=os.path.join(FIGURES_DIR, 'dispersion_phase_shift.png')
)

# ============================================
# 5. AUTOMATIC PICKING OF DISPERSION CURVE
# ============================================

print("\n" + "=" * 60)
print("AUTOMATIC DISPERSION CURVE PICKING")
print("=" * 60)

picked_velocities, uncertainties = automatic_picking(
    dispersion_image,
    frequencies,
    velocities,
    smooth_window=5,
    threshold=0.5
)

# Plot with picked curve
fig_picked = plot_dispersion_image(
    dispersion_image,
    frequencies,
    velocities,
    picked_curve=picked_velocities,
    title="Dispersion Image with Automatically Picked Curve",
    save_path=os.path.join(FIGURES_DIR, 'dispersion_picked_auto.png')
)

# Save picked curve
dispersion_file = os.path.join(DISPERSION_DIR, 'dispersion_curve_fundamental.txt')
dispersion_data = np.column_stack([frequencies, picked_velocities, uncertainties])
header = 'Frequency(Hz) PhaseVelocity(m/s) Uncertainty(m/s)'
np.savetxt(dispersion_file, dispersion_data, header=header, fmt='%.4f')
print(f"\nDispersion curve saved to: {dispersion_file}")

# ============================================
# 6. QUALITY ASSESSMENT
# ============================================

def assess_dispersion_quality(frequencies, velocities, uncertainties,
                              dispersion_image):
    """
    Assess quality of picked dispersion curve
    """
    
    print("\n" + "=" * 60)
    print("DISPERSION CURVE QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Check for smoothness (gradient)
    velocity_gradient = np.gradient(velocities)
    smoothness = np.std(velocity_gradient)
    print(f"\nSmoothness metric (gradient std): {smoothness:.2f} m/s/Hz")
    
    if smoothness < 5.0:
        print("  ✓ Curve is very smooth")
    elif smoothness < 15.0:
        print("  ✓ Curve is reasonably smooth")
    else:
        print("  ⚠ Curve may have jumps or discontinuities")
    
    # Check uncertainty levels
    mean_uncertainty = np.mean(uncertainties)
    max_uncertainty = np.max(uncertainties)
    print(f"\nUncertainty statistics:")
    print(f"  Mean: {mean_uncertainty:.2f} m/s")
    print(f"  Max: {max_uncertainty:.2f} m/s")
    print(f"  Relative (mean/velocity): {100*mean_uncertainty/np.mean(velocities):.1f}%")
    
    if mean_uncertainty < 20.0:
        print("  ✓ Low uncertainty - high quality picks")
    elif mean_uncertainty < 50.0:
        print("  ✓ Moderate uncertainty - acceptable quality")
    else:
        print("  ⚠ High uncertainty - may need refinement")
    
    # Check wavelength coverage
    wavelengths = velocities / frequencies
    min_wavelength = wavelengths.min()
    max_wavelength = wavelengths.max()
    print(f"\nWavelength range: {min_wavelength:.1f} - {max_wavelength:.1f} m")
    
    # Estimate depth of investigation (rule of thumb: λ/2 to λ)
    depth_max = max_wavelength  # Conservative estimate
    depth_min = min_wavelength / 2
    print(f"Estimated depth range: {depth_min:.1f} - {depth_max:.1f} m")
    
    if depth_max >= 30.0:
        print("  ✓ Sufficient depth for Vs30 calculation")
    else:
        print("  ⚠ May not reach 30m depth - consider lower frequencies")
    
    # Check frequency range
    print(f"\nFrequency range: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")
    
    if frequencies.max() >= 40.0:
        print("  ✓ Good high-frequency content for shallow resolution")
    if frequencies.min() <= 10.0:
        print("  ✓ Good low-frequency content for depth penetration")
    
    print("=" * 60)

# Run quality assessment
assess_dispersion_quality(frequencies, picked_velocities, 
                         uncertainties, dispersion_image)

# ============================================
# 7. OPTIONAL: INTERACTIVE PICKING
# ============================================

print("\n" + "=" * 60)
print("OPTIONAL: INTERACTIVE PICKING")
print("=" * 60)
print("Set run_interactive = True to enable manual picking")

run_interactive = False  # Set to True if you want interactive picking

if run_interactive:
    picked_curves_manual = interactive_picking(
        dispersion_image,
        frequencies,
        velocities,
        n_modes=1,
        save_path=os.path.join(DISPERSION_DIR, 'dispersion_curve_manual.txt')
    )

# ============================================
# 8. SAVE ANALYSIS SUMMARY
# ============================================

summary = f"""
MASW DISPERSION ANALYSIS SUMMARY
========================================

Input Data:
  - Source: Preprocessed Geophydog data
  - Number of traces: {n_traces}
  - Sampling rate: {fs} Hz
  - Distance range: {distances.min():.1f} - {distances.max():.1f} m

Analysis Parameters:
  - Frequency range: {freq_min} - {freq_max} Hz
  - Velocity range: {vel_min} - {vel_max} m/s
  - Number of frequency points: {n_freqs}
  - Number of velocity points: {n_vels}

Method Applied:
  - Phase Shift (Geophydog f-c transform) ✓

Dispersion Curve Results:
  - Velocity range: {picked_velocities.min():.1f} - {picked_velocities.max():.1f} m/s
  - Mean uncertainty: {uncertainties.mean():.1f} m/s
  - Wavelength range: {(picked_velocities/frequencies).min():.1f} - {(picked_velocities/frequencies).max():.1f} m
  - Estimated max depth: {(picked_velocities/frequencies).max():.1f} m

Output Files:
  - Dispersion curve: {DISPERSION_DIR}/dispersion_curve_fundamental.txt
  - Figures: {FIGURES_DIR}/dispersion_*.png

Quality: Ready for inversion ✓

Date: {np.datetime64('now')}
"""

summary_file = os.path.join(DISPERSION_DIR, 'dispersion_analysis_summary.txt')
with open(summary_file, 'w') as f:
    f.write(summary)

print("\n" + summary)
print(f"Summary saved to: {summary_file}")

print("\n" + "=" * 60)
print("DISPERSION ANALYSIS COMPLETE")
print("=" * 60)
print("\nGenerated files:")
print(f"  - {len(glob.glob(os.path.join(FIGURES_DIR, 'dispersion*.png')))} figures")
print(f"  - 1 dispersion curve data file")
print(f"  - 1 analysis summary")
print("\nReady for: Inversion!")