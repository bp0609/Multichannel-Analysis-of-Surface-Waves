# notebooks/03_dispersion_analysis.ipynb
# Or: code/dispersion_analysis/extract_dispersion.py

"""
Phase 4: Dispersion Analysis
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
from config import PROCESSED_DATA_DIR, DISPERSION_DIR, FIGURES_DIR, DISTANCE_CORRECTION_FACTOR
from dispersion_analysis.phase_shift import (
    phase_shift_transform,
    fk_transform,
    slant_stack_transform,
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
sac_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.sac")))
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
freq_min = 5.0
freq_max = 50.0
n_freqs = 100
frequencies = np.linspace(freq_min, freq_max, n_freqs)

# Velocity range for analysis (m/s)
vel_min = 100.0
vel_max = 1000.0
n_vels = 200
velocities = np.linspace(vel_min, vel_max, n_vels)

print(f"\nAnalysis parameters:")
print(f"  Frequency range: {freq_min} - {freq_max} Hz ({n_freqs} points)")
print(f"  Velocity range: {vel_min} - {vel_max} m/s ({n_vels} points)")

# ============================================
# 4. COMPUTE DISPERSION IMAGES - PHASE SHIFT METHOD
# ============================================

print("\n" + "=" * 60)
print("METHOD 1: PHASE SHIFT (CYLINDRICAL WAVEFRONT)")
print("=" * 60)

dispersion_ps = phase_shift_transform(
    data=data,
    dt=dt,
    offsets=distances,
    frequencies=frequencies,
    velocities=velocities,
    wave_type='cylindrical'
)

# Plot phase shift result
fig_ps = plot_dispersion_image(
    dispersion_ps,
    frequencies,
    velocities,
    title="Dispersion Image - Phase Shift Method (Cylindrical)",
    save_path=os.path.join(FIGURES_DIR, 'dispersion_phase_shift.png')
)

# ============================================
# 5. COMPUTE DISPERSION IMAGES - F-K TRANSFORM
# ============================================

print("\n" + "=" * 60)
print("METHOD 2: F-K TRANSFORM")
print("=" * 60)

# Calculate receiver spacing
dx = np.median(np.diff(distances))

dispersion_fk, freqs_fk, vels_fk = fk_transform(
    data=data,
    dt=dt,
    dx=dx,
    freq_range=(freq_min, freq_max),
    vel_range=(vel_min, vel_max)
)

# Plot f-k result
fig_fk = plot_dispersion_image(
    dispersion_fk,
    freqs_fk,
    vels_fk,
    title="Dispersion Image - F-K Transform",
    save_path=os.path.join(FIGURES_DIR, 'dispersion_fk.png')
)

# ============================================
# 6. COMPUTE DISPERSION IMAGES - SLANT STACK
# ============================================

print("\n" + "=" * 60)
print("METHOD 3: SLANT STACK (TAU-P)")
print("=" * 60)

dispersion_ss = slant_stack_transform(
    data=data,
    dt=dt,
    offsets=distances,
    frequencies=frequencies,
    velocities=velocities
)

# Plot slant-stack result
fig_ss = plot_dispersion_image(
    dispersion_ss,
    frequencies,
    velocities,
    title="Dispersion Image - Slant Stack (Tau-P)",
    save_path=os.path.join(FIGURES_DIR, 'dispersion_slant_stack.png')
)

# ============================================
# 7. COMPARE ALL METHODS
# ============================================

def compare_methods(disp_ps, disp_fk, disp_ss, frequencies, velocities, 
                    save_path=None):
    """Compare dispersion images from different methods"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = [
        (disp_ps, "Phase Shift"),
        (disp_fk, "F-K Transform"),
        (disp_ss, "Slant Stack")
    ]
    
    extent = [velocities.min(), velocities.max(), 
              frequencies.min(), frequencies.max()]
    
    for ax, (disp, title) in zip(axes, methods):
        im = ax.imshow(disp, aspect='auto', origin='lower',
                      extent=extent, cmap='jet', interpolation='bilinear')
        ax.set_xlabel('Phase Velocity (m/s)', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', color='white')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle('Comparison of Dispersion Extraction Methods', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

print("\n" + "=" * 60)
print("COMPARING ALL METHODS")
print("=" * 60)

fig_compare = compare_methods(
    dispersion_ps,
    dispersion_fk,
    dispersion_ss,
    frequencies,
    velocities,
    save_path=os.path.join(FIGURES_DIR, 'dispersion_comparison.png')
)

# ============================================
# 8. AUTOMATIC PICKING OF DISPERSION CURVE
# ============================================

print("\n" + "=" * 60)
print("AUTOMATIC DISPERSION CURVE PICKING")
print("=" * 60)

# Use phase shift method (typically most reliable)
picked_velocities, uncertainties = automatic_picking(
    dispersion_ps,
    frequencies,
    velocities,
    smooth_window=5,
    threshold=0.5
)

# Plot with picked curve
fig_picked = plot_dispersion_image(
    dispersion_ps,
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
# 9. PLOT DISPERSION CURVE IN DIFFERENT FORMATS
# ============================================

def plot_dispersion_curve_formats(frequencies, velocities, uncertainties,
                                   save_path=None):
    """
    Plot dispersion curve in multiple formats
    """
    
    # Calculate wavelength
    wavelengths = velocities / frequencies
    
    # Calculate period
    periods = 1.0 / frequencies
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Velocity vs Frequency
    ax = axes[0, 0]
    ax.errorbar(frequencies, velocities, yerr=uncertainties,
                fmt='o-', color='blue', capsize=3, capthick=1,
                markersize=4, linewidth=1.5, label='Fundamental Mode')
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontweight='bold')
    ax.set_title('Dispersion Curve: Velocity vs Frequency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Velocity vs Wavelength
    ax = axes[0, 1]
    ax.errorbar(wavelengths, velocities, yerr=uncertainties,
                fmt='o-', color='green', capsize=3, capthick=1,
                markersize=4, linewidth=1.5)
    ax.set_xlabel('Wavelength (m)', fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontweight='bold')
    ax.set_title('Dispersion Curve: Velocity vs Wavelength', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocity vs Period
    ax = axes[1, 0]
    ax.errorbar(periods, velocities, yerr=uncertainties,
                fmt='o-', color='red', capsize=3, capthick=1,
                markersize=4, linewidth=1.5)
    ax.set_xlabel('Period (s)', fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontweight='bold')
    ax.set_title('Dispersion Curve: Velocity vs Period', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Slowness vs Frequency
    ax = axes[1, 1]
    slowness = 1000.0 / velocities  # Convert to s/km
    slowness_unc = 1000.0 * uncertainties / velocities**2
    ax.errorbar(frequencies, slowness, yerr=slowness_unc,
                fmt='o-', color='purple', capsize=3, capthick=1,
                markersize=4, linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Slowness (s/km)', fontweight='bold')
    ax.set_title('Dispersion Curve: Slowness vs Frequency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Dispersion Curve - Multiple Representations', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

print("\n" + "=" * 60)
print("PLOTTING DISPERSION CURVE IN MULTIPLE FORMATS")
print("=" * 60)

fig_formats = plot_dispersion_curve_formats(
    frequencies,
    picked_velocities,
    uncertainties,
    save_path=os.path.join(FIGURES_DIR, 'dispersion_curve_formats.png')
)

# ============================================
# 10. QUALITY ASSESSMENT
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
                         uncertainties, dispersion_ps)

# ============================================
# 11. OPTIONAL: INTERACTIVE PICKING
# ============================================

print("\n" + "=" * 60)
print("OPTIONAL: INTERACTIVE PICKING")
print("=" * 60)
print("Set run_interactive = True to enable manual picking")

run_interactive = False  # Set to True if you want interactive picking

if run_interactive:
    picked_curves_manual = interactive_picking(
        dispersion_ps,
        frequencies,
        velocities,
        n_modes=1,
        save_path=os.path.join(DISPERSION_DIR, 'dispersion_curve_manual.txt')
    )

# ============================================
# 12. SAVE ANALYSIS SUMMARY
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

Methods Applied:
  1. Phase Shift (Cylindrical) ✓
  2. F-K Transform ✓
  3. Slant Stack (Tau-P) ✓

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
print("PHASE 4 COMPLETE: Dispersion Analysis")
print("=" * 60)
print("\nGenerated files:")
print(f"  - {len(glob.glob(os.path.join(FIGURES_DIR, 'dispersion*.png')))} figures")
print(f"  - 1 dispersion curve data file")
print(f"  - 1 analysis summary")
print("\nReady for Phase 5: Inversion!")