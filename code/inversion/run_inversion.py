"""
Inversion for Vs Profile
Invert dispersion curve to obtain shear-wave velocity profile
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project paths
sys.path.append('..')
from config import DISPERSION_DIR, FIGURES_DIR
from inversion.forward_model import (
    LayeredEarthModel, compute_dispersion_curve, 
    plot_model_and_dispersion
)
from inversion.initial_model import (
    create_initial_model, create_initial_model_from_dispersion,
    define_parameter_bounds
)
from inversion.least_square import invert_least_squares
from inversion.global_search import (
    monte_carlo_inversion, analyze_monte_carlo_results
)
from inversion.hybrid import hybrid_inversion

# Create results directory
os.makedirs(os.path.join(FIGURES_DIR, 'inversion'), exist_ok=True)

# ============================================
# 1. LOAD OBSERVED DISPERSION CURVE
# ============================================

print("=" * 60)
print("LOADING OBSERVED DISPERSION CURVE")
print("=" * 60)

# Load dispersion curve
dispersion_file = os.path.join(DISPERSION_DIR, 'dispersion_curve_fundamental.txt')
data = np.loadtxt(dispersion_file)

frequencies = data[:, 0]
observed_velocities = data[:, 1]
uncertainties = data[:, 2]

print(f"Loaded dispersion curve:")
print(f"  Frequency range: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")
print(f"  Velocity range: {observed_velocities.min():.1f} - {observed_velocities.max():.1f} m/s")
print(f"  Mean uncertainty: {uncertainties.mean():.1f} m/s")

# Plot observed dispersion curve
fig, ax = plt.subplots(figsize=(14, 7))

# Plot uncertainty as filled region (cleaner than error bars on every point)
ax.fill_between(frequencies, 
                observed_velocities - uncertainties, 
                observed_velocities + uncertainties,
                alpha=0.3, color='red', label='Uncertainty (±1σ)')

# Plot the dispersion curve line
ax.plot(frequencies, observed_velocities, 'r-', linewidth=2.5, 
        label='Observed Dispersion Curve', zorder=3)

# Plot markers at reduced frequency (every 10th point) for clarity
marker_step = 10
ax.plot(frequencies[::marker_step], observed_velocities[::marker_step], 
        'ro', markersize=7, markeredgewidth=1.5, markeredgecolor='darkred',
        zorder=4)

# Formatting
ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
ax.set_ylabel('Phase Velocity (m/s)', fontsize=13, fontweight='bold')
ax.set_title('Observed Fundamental Mode Dispersion Curve', 
             fontsize=15, fontweight='bold', pad=15)

# Adjust x-axis for better spacing
ax.set_xlim(frequencies.min() - 1, frequencies.max() + 1)
ax.set_ylim(observed_velocities.min() - 50, observed_velocities.max() + 50)

# Grid and legend
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Add minor ticks for better readability
ax.minorticks_on()
ax.grid(True, which='minor', alpha=0.2, linestyle=':')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'inversion', 'observed_dispersion.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. CREATE INITIAL MODEL
# ============================================

print("\n" + "=" * 60)
print("CREATING INITIAL MODEL")
print("=" * 60)

# Define number of layers
n_layers = 5  # 4 layers + half-space

# Method 1: Simple linear model
initial_model_linear = create_initial_model(
    n_layers=n_layers,
    method='linear',
    vs_range=(200, 600),
    thickness_range=(5, 15)
)

print("\nInitial Model (Linear):")
print(initial_model_linear)

# Method 2: Estimated from dispersion curve
initial_model_estimated = create_initial_model_from_dispersion(
    frequencies, observed_velocities,
    n_layers=n_layers,
    depth_max=50
)

print("\nInitial Model (From Dispersion):")
print(initial_model_estimated)

# Choose which to use
initial_model = initial_model_estimated

# Plot initial model
plot_model_and_dispersion(
    initial_model, frequencies, observed_velocities,
    save_path=os.path.join(FIGURES_DIR, 'inversion', 'initial_model.png')
)

# ============================================
# 3. DEFINE PARAMETER BOUNDS
# ============================================

print("\n" + "=" * 60)
print("DEFINING PARAMETER BOUNDS")
print("=" * 60)

bounds = define_parameter_bounds(
    n_layers=initial_model.n_layers,  # Use actual number of layers from model
    vs_min=100,
    vs_max=1500,
    h_min=2,
    h_max=40
)

print("\nParameter bounds:")
print(f"  Vs: {bounds['vs'][0][0]} - {bounds['vs'][0][1]} m/s")
print(f"  Thickness: {bounds['thickness'][0][0]} - {bounds['thickness'][0][1]} m")

# ============================================
# 4. RUN INVERSIONS
# ============================================

# ----------------------------------------
# 4A. Least-Squares Inversion
# ----------------------------------------

print("\n" + "=" * 60)
print("RUNNING LEAST-SQUARES INVERSION")
print("=" * 60)

model_ls, result_ls, rms_ls = invert_least_squares(
    frequencies, observed_velocities,
    initial_model=initial_model,
    bounds=bounds,
    uncertainties=uncertainties,
    max_iterations=100,
    verbose=True
)

# Plot least-squares result
plot_model_and_dispersion(
    model_ls, frequencies, observed_velocities,
    save_path=os.path.join(FIGURES_DIR, 'inversion', 'result_least_squares.png')
)

# ----------------------------------------
# 4B. Monte Carlo Inversion
# ----------------------------------------

print("\n" + "=" * 60)
print("RUNNING MONTE CARLO INVERSION")
print("=" * 60)

model_mc, misfit_mc, all_models_mc, all_misfits_mc = monte_carlo_inversion(
    frequencies, observed_velocities,
    n_models=1000,
    n_layers=initial_model.n_layers,
    bounds=bounds,
    uncertainties=uncertainties,
    verbose=True
)

# Analyze results
acceptable_models, vs_ranges = analyze_monte_carlo_results(
    all_models_mc, all_misfits_mc, threshold_percentile=10
)

# Plot Monte Carlo result
plot_model_and_dispersion(
    model_mc, frequencies, observed_velocities,
    save_path=os.path.join(FIGURES_DIR, 'inversion', 'result_monte_carlo.png')
)

# ----------------------------------------
# 4C. Hybrid Inversion
# ----------------------------------------

print("\n" + "=" * 60)
print("RUNNING HYBRID INVERSION")
print("=" * 60)

model_hybrid, model_hybrid_mc, rms_hybrid = hybrid_inversion(
    frequencies, observed_velocities,
    n_layers=initial_model.n_layers,
    n_monte_carlo=500,
    bounds=bounds,
    uncertainties=uncertainties,
    verbose=True
)

# Plot hybrid result
plot_model_and_dispersion(
    model_hybrid, frequencies, observed_velocities,
    save_path=os.path.join(FIGURES_DIR, 'inversion', 'result_hybrid.png')
)

# ============================================
# 5. COMPARE ALL RESULTS
# ============================================

def compare_inversion_results(models, labels, frequencies, observed_velocities,
                             save_path=None):
    """
    Compare results from different inversion methods
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot 1: Vs profiles
    ax = axes[0]
    
    for model, label, color in zip(models, labels, colors):
        depths, vs_profile = model.get_depth_array(dz=0.5)
        ax.plot(vs_profile, depths, color=color, linewidth=2.5, 
                label=f'{label} (Vs30={model.calculate_vs30():.0f} m/s)', alpha=0.8)
    
    ax.axhline(30, color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(0.98, 30, '  30m', verticalalignment='center',
            horizontalalignment='right', fontsize=10)
    
    ax.set_xlabel('Vs (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('Shear Wave Velocity Profiles', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(50, 0)
    
    # Plot 2: Dispersion curves
    ax = axes[1]
    
    # Observed
    ax.errorbar(frequencies, observed_velocities, yerr=None,
                fmt='ko', markersize=8, label='Observed', alpha=0.7, zorder=10)
    
    # Theoretical curves
    for model, label, color in zip(models, labels, colors):
        theoretical = compute_dispersion_curve(model, frequencies)
        ax.plot(frequencies, theoretical, color=color, linewidth=2.5,
                label=label, marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Dispersion Curves', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    fig.suptitle('Comparison of Inversion Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

print("\n" + "=" * 60)
print("COMPARING ALL INVERSION RESULTS")
print("=" * 60)

models_compare = [initial_model, model_ls, model_mc, model_hybrid]
labels_compare = ['Initial', 'Least-Squares', 'Monte Carlo', 'Hybrid']

fig_compare = compare_inversion_results(
    models_compare, labels_compare, frequencies, observed_velocities,
    save_path=os.path.join(FIGURES_DIR, 'inversion', 'comparison_all.png')
)

# ============================================
# 6. ASSESS UNCERTAINTY WITH MONTE CARLO
# ============================================

def plot_uncertainty_envelope(frequencies, observed_velocities, 
                             acceptable_models, save_path=None):
    """
    Plot uncertainty envelope from acceptable models
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Compute dispersion curves for all acceptable models
    n_models = len(acceptable_models)
    n_freqs = len(frequencies)
    
    all_vs_profiles = []
    all_dispersion = np.zeros((n_models, n_freqs))
    
    for i, model in enumerate(acceptable_models):
        all_dispersion[i, :] = compute_dispersion_curve(model, frequencies)
        depths, vs_profile = model.get_depth_array(dz=0.5)
        all_vs_profiles.append((depths, vs_profile))
    
    # Plot 1: Vs profiles with envelope
    ax = axes[0]
    
    # Plot all acceptable profiles in light gray
    for depths, vs_profile in all_vs_profiles:
        ax.plot(vs_profile, depths, 'gray', alpha=0.1, linewidth=0.5)
    
    # Calculate and plot mean and std
    # Need to interpolate to common depth grid first
    max_depth = 50
    common_depths = np.linspace(0, max_depth, 200)
    
    vs_at_depths = np.zeros((n_models, len(common_depths)))
    for i, (depths, vs_profile) in enumerate(all_vs_profiles):
        vs_at_depths[i, :] = np.interp(common_depths, depths, vs_profile)
    
    mean_vs = np.mean(vs_at_depths, axis=0)
    std_vs = np.std(vs_at_depths, axis=0)
    
    ax.plot(mean_vs, common_depths, 'b-', linewidth=3, label='Mean')
    ax.fill_betweenx(common_depths, mean_vs - std_vs, mean_vs + std_vs,
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax.axhline(30, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax.set_xlabel('Vs (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('Vs Profile Uncertainty', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(max_depth, 0)
    
    # Plot 2: Dispersion curve uncertainty
    ax = axes[1]
    
    # Observed
    ax.plot(frequencies, observed_velocities, 'ko', markersize=8,
            label='Observed', zorder=10)
    
    # All acceptable curves
    for i in range(n_models):
        ax.plot(frequencies, all_dispersion[i, :], 'gray',
                alpha=0.1, linewidth=0.5)
    
    # Mean and std
    mean_disp = np.mean(all_dispersion, axis=0)
    std_disp = np.std(all_dispersion, axis=0)
    
    ax.plot(frequencies, mean_disp, 'b-', linewidth=3, label='Mean')
    ax.fill_between(frequencies, mean_disp - std_disp, mean_disp + std_disp,
                    alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Dispersion Curve Uncertainty', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    fig.suptitle(f'Uncertainty Assessment ({n_models} Acceptable Models)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

print("\n" + "=" * 60)
print("PLOTTING UNCERTAINTY ENVELOPE")
print("=" * 60)

fig_uncertainty = plot_uncertainty_envelope(
    frequencies, observed_velocities, acceptable_models,
    save_path=os.path.join(FIGURES_DIR, 'inversion', 'uncertainty_envelope.png')
)

# ============================================
# 7. SAVE FINAL MODEL
# ============================================

print("\n" + "=" * 60)
print("SAVING FINAL MODEL")
print("=" * 60)

# Choose best model (hybrid typically best)
final_model = model_hybrid

# Save model to file
model_file = os.path.join(DISPERSION_DIR, 'vs_profile_final.txt')

# Prepare data
model_data = np.column_stack([
    range(1, final_model.n_layers + 1),
    final_model.thickness,
    final_model.vs,
    final_model.vp,
    final_model.rho
])

header = 'Layer Thickness(m) Vs(m/s) Vp(m/s) Rho(g/cm3)'
np.savetxt(model_file, model_data, header=header, 
           fmt=['%d', '%.2f', '%.1f', '%.1f', '%.2f'])

print(f"Model saved to: {model_file}")

# Save summary
summary = f"""
MASW INVERSION SUMMARY
========================================

Observed Data:
  - Frequency range: {frequencies.min():.1f} - {frequencies.max():.1f} Hz
  - Number of data points: {len(frequencies)}
  - Mean uncertainty: {uncertainties.mean():.1f} m/s

Inversion Results:
  - Number of layers: {initial_model.n_layers}
  
  Least-Squares:
    - RMS error: {rms_ls:.2f} m/s
    - Vs30: {model_ls.calculate_vs30():.1f} m/s
  
  Monte Carlo:
    - Best RMS error: {misfit_mc:.2f} m/s
    - Vs30: {model_mc.calculate_vs30():.1f} m/s
    - Models tested: {len(all_models_mc)}
    - Acceptable models: {len(acceptable_models)}
  
  Hybrid (FINAL):
    - RMS error: {rms_hybrid:.2f} m/s
    - Vs30: {model_hybrid.calculate_vs30():.1f} m/s

Final Model:
{final_model}

Output Files:
  - Model: {model_file}
  - Figures: {FIGURES_DIR}/inversion/

Date: {np.datetime64('now')}
"""

summary_file = os.path.join(DISPERSION_DIR, 'inversion_summary.txt')
with open(summary_file, 'w') as f:
    f.write(summary)

print("\n" + summary)
print(f"Summary saved to: {summary_file}")

print("\n" + "=" * 60)
print("INVERSION COMPLETE: Vs Profile Obtained")
print("=" * 60)
print("\nReady for: Vs30 Calculation and Site Classification")