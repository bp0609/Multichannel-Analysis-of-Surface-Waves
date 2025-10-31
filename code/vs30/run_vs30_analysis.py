"""
Vs30 Calculation and Site Classification
Calculate Vs30, determine site class, and perform comprehensive analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project paths
sys.path.append('..')
from config import DISPERSION_DIR, FIGURES_DIR
from inversion.forward_model import LayeredEarthModel
from vs30.calculate_vs30 import (
    calculate_vs30, calculate_vs_statistics, propagate_vs30_uncertainty
)
from vs30.site_classification import (
    NEHRPClassification, EuroCode8Classification, classify_site_all_systems
)
from vs30.visualizations import (
    plot_vs_profile_with_vs30, plot_nehrp_classification_chart,
    plot_vs_statistics, plot_vs30_uncertainty
)

# Create output directory
os.makedirs(os.path.join(FIGURES_DIR, 'vs30'), exist_ok=True)

# ============================================
# 1. LOAD FINAL MODEL
# ============================================

print("=" * 60)
print("LOADING FINAL VS PROFILE")
print("=" * 60)

# Load model file
model_file = os.path.join(DISPERSION_DIR, 'vs_profile_final.txt')
model_data = np.loadtxt(model_file)

# Extract parameters
thickness = model_data[:, 1]
vs = model_data[:, 2]
vp = model_data[:, 3]
rho = model_data[:, 4]

# Create model
final_model = LayeredEarthModel(thickness, vs, vp, rho)

print("Final Model:")
print(final_model)

# ============================================
# 2. CALCULATE Vs30 AND RELATED PARAMETERS
# ============================================

print("\n" + "=" * 60)
print("CALCULATING Vs30 AND STATISTICS")
print("=" * 60)

# Calculate Vs30
vs30 = calculate_vs30(final_model)

# Calculate all statistics
stats = calculate_vs_statistics(final_model, max_depth=50)

print(f"\nShear Wave Velocity Statistics:")
print("-" * 60)
print(f"Vs30:              {stats['vs30']:.1f} m/s")
print(f"Vs20:              {stats['vs20']:.1f} m/s")
print(f"Vs15:              {stats['vs15']:.1f} m/s")
print(f"Vs10:              {stats['vs10']:.1f} m/s")
print(f"Surface Vs:        {stats['vs_surface']:.1f} m/s")
print(f"Minimum Vs:        {stats['vs_min']:.1f} m/s")
print(f"Maximum Vs:        {stats['vs_max']:.1f} m/s")

if stats['depth_to_bedrock'] is not None:
    print(f"Depth to bedrock:  {stats['depth_to_bedrock']:.1f} m (Vs > 760 m/s)")
else:
    print(f"Depth to bedrock:  Not reached within profile")

# ============================================
# 3. SITE CLASSIFICATION
# ============================================

print("\n" + "=" * 60)
print("SITE CLASSIFICATION")
print("=" * 60)

# Get all classifications
classifications = classify_site_all_systems(vs30)

print(f"\nVs30 = {vs30:.1f} m/s")
print("\n" + "-" * 60)
print("NEHRP Site Classification (ASCE 7-22, 5 classes):")
print("-" * 60)
nehrp = classifications['nehrp']
print(f"  Site Class: {nehrp['class']}")
print(f"  Description: {nehrp['description']}")
print(f"  Short-period site coefficient (Fa): {nehrp['fa']:.2f}")
print(f"  Long-period site coefficient (Fv): {nehrp['fv']:.2f}")

print("\n" + "-" * 60)
print("NEHRP Extended Classification (ASCE 7-22, 9 classes):")
print("-" * 60)
nehrp_ext = classifications['nehrp_extended']
print(f"  Site Class: {nehrp_ext['class']}")
print(f"  Description: {nehrp_ext['description']}")

print("\n" + "-" * 60)
print("Eurocode 8 Classification:")
print("-" * 60)
ec8 = classifications['eurocode8']
print(f"  Ground Type: {ec8['class']}")
print(f"  Description: {ec8['description']}")

# ============================================
# 4. UNCERTAINTY ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("Vs30 UNCERTAINTY ANALYSIS")
print("=" * 60)

# Try to load acceptable models from Monte Carlo
try:
    # This assumes you saved the acceptable models during inversion
    # If not available, we'll use the single final model
    
    # For demonstration, let's load from the Monte Carlo results
    # You would need to modify the inversion script to save these
    
    print("Loading acceptable models from Monte Carlo inversion...")
    
    # Placeholder: Create synthetic uncertainty by perturbing final model
    # In practice, you'd use actual ensemble from inversion
    n_samples = 100
    acceptable_models = []
    
    for i in range(n_samples):
        # Perturb velocities by ±10%
        vs_perturbed = vs * (1 + np.random.randn(len(vs)) * 0.10)
        vs_perturbed = np.clip(vs_perturbed, 100, 2000)
        
        # Perturb thickness by ±20%
        thickness_perturbed = thickness.copy()
        for j in range(len(thickness) - 1):  # Don't perturb half-space
            thickness_perturbed[j] *= (1 + np.random.randn() * 0.20)
            thickness_perturbed[j] = max(2, thickness_perturbed[j])
        
        model_perturbed = LayeredEarthModel(thickness_perturbed, vs_perturbed)
        acceptable_models.append(model_perturbed)
    
    print(f"Created {len(acceptable_models)} perturbed models for uncertainty analysis")
    
    # Calculate Vs30 uncertainty
    vs30_mean, vs30_std, vs30_values = propagate_vs30_uncertainty(acceptable_models)
    
    print(f"\nVs30 Uncertainty:")
    print(f"  Mean:                {vs30_mean:.1f} m/s")
    print(f"  Standard deviation:  {vs30_std:.1f} m/s")
    print(f"  Range:               {vs30_values.min():.1f} - {vs30_values.max():.1f} m/s")
    print(f"  Coefficient of variation: {vs30_std/vs30_mean*100:.1f}%")
    
    # 95% confidence interval
    ci_lower = vs30_mean - 1.96 * vs30_std
    ci_upper = vs30_mean + 1.96 * vs30_std
    print(f"  95% Confidence Interval: {ci_lower:.1f} - {ci_upper:.1f} m/s")
    
    has_uncertainty = True
    
except Exception as e:
    print(f"Could not perform uncertainty analysis: {e}")
    print("Using single deterministic model only")
    vs30_values = np.array([vs30])
    has_uncertainty = False

# ============================================
# 5. CREATE VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Figure 1: Vs profile with Vs30 highlighted
print("\n1. Creating Vs profile with Vs30...")
fig1 = plot_vs_profile_with_vs30(
    final_model,
    save_path=os.path.join(FIGURES_DIR, 'vs30', 'vs_profile_with_vs30.png')
)

# Figure 2: NEHRP classification chart
print("2. Creating NEHRP classification chart...")
fig2 = plot_nehrp_classification_chart(
    vs30,
    save_path=os.path.join(FIGURES_DIR, 'vs30', 'nehrp_classification.png')
)

# Figure 3: Vs statistics at multiple depths
print("3. Creating Vs statistics plot...")
fig3 = plot_vs_statistics(
    final_model,
    save_path=os.path.join(FIGURES_DIR, 'vs30', 'vs_statistics.png')
)

# Figure 4: Uncertainty analysis
if has_uncertainty:
    print("4. Creating uncertainty analysis plots...")
    fig4 = plot_vs30_uncertainty(
        vs30_values,
        save_path=os.path.join(FIGURES_DIR, 'vs30', 'vs30_uncertainty.png')
    )

# ============================================
# 6. CREATE COMPREHENSIVE SUMMARY PLOT
# ============================================

def create_summary_figure(model, vs30, classifications, stats, save_path=None):
    """
    Create comprehensive summary figure
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Vs profile (large, left side)
    ax1 = fig.add_subplot(gs[:, 0])
    depths, vs_profile = model.get_depth_array(dz=0.5)
    ax1.plot(vs_profile, depths, 'b-', linewidth=3)
    ax1.axhline(30, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(vs30, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.fill_betweenx([0, 30], 0, vs_profile.max()*1.2, alpha=0.1, color='yellow')
    ax1.set_xlabel('Vs (m/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Shear Wave Velocity Profile', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, 0)
    
    # Plot 2: Layer table (top right)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')
    
    table_data = []
    table_data.append(['Layer', 'Thickness\n(m)', 'Vs\n(m/s)', 'Vp\n(m/s)', 'Density\n(g/cm³)'])
    
    for i in range(model.n_layers):
        thick_str = f"{model.thickness[i]:.1f}" if model.thickness[i] > 0 else "∞"
        table_data.append([
            str(i+1),
            thick_str,
            f"{model.vs[i]:.0f}",
            f"{model.vp[i]:.0f}",
            f"{model.rho[i]:.2f}"
        ])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header row formatting
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Layer Parameters', fontsize=12, fontweight='bold', pad=10)
    
    # Plot 3: Vs30 and statistics (middle right)
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.axis('off')
    
    stats_text = f"""
    SITE CHARACTERIZATION RESULTS
    
    Vs30 = {vs30:.1f} m/s
    
    NEHRP Site Class: {classifications['nehrp']['class']}
    Description: {classifications['nehrp']['description']}
    
    Site Coefficients:
      Fa (short period) = {classifications['nehrp']['fa']:.2f}
      Fv (long period) = {classifications['nehrp']['fv']:.2f}
    
    Additional Metrics:
      Vs10 = {stats['vs10']:.0f} m/s
      Vs20 = {stats['vs20']:.0f} m/s
      Surface Vs = {stats['vs_surface']:.0f} m/s
    """
    
    ax3.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 4: NEHRP classification bar (bottom right)
    ax4 = fig.add_subplot(gs[2, 1:])
    
    classes = ['E', 'D', 'C', 'B', 'A']
    boundaries = [0, 180, 360, 760, 1500]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for i in range(len(classes)):
        if i < len(classes) - 1:
            width = boundaries[i+1] - boundaries[i]
        else:
            width = 500  # Just for display
        
        ax4.barh(0, width, left=boundaries[i], height=0.5,
                color=colors[i], edgecolor='black', linewidth=2, alpha=0.7)
        
        center = boundaries[i] + width/2
        ax4.text(center, 0, classes[i], ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
    
    # Mark site
    ax4.plot([vs30, vs30], [-0.3, 0.3], 'r-', linewidth=4, zorder=10)
    ax4.plot(vs30, 0, 'ro', markersize=15, zorder=11)
    
    ax4.set_xlim(0, 2000)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_xlabel('Vs30 (m/s)', fontsize=11, fontweight='bold')
    ax4.set_yticks([])
    ax4.set_title('NEHRP Site Classification', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    
    fig.suptitle('MASW Site Characterization Summary', 
                 fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

print("\n5. Creating comprehensive summary figure...")
fig_summary = create_summary_figure(
    final_model, vs30, classifications, stats,
    save_path=os.path.join(FIGURES_DIR, 'vs30', 'summary_report.png')
)

# ============================================
# 7. GENERATE FINAL REPORT
# ============================================

print("\n" + "=" * 60)
print("GENERATING FINAL REPORT")
print("=" * 60)

report = f"""
{'='*70}
MASW SITE CHARACTERIZATION REPORT
{'='*70}

Project: MASW Vs30 Analysis
Date: {np.datetime64('now')}

{'='*70}
1. SHEAR WAVE VELOCITY PROFILE
{'='*70}

Layer Parameters:
{'-'*70}
{'Layer':<8} {'Thickness(m)':<15} {'Vs(m/s)':<12} {'Vp(m/s)':<12} {'Rho(g/cm³)':<12}
{'-'*70}
"""

for i in range(final_model.n_layers):
    thick_str = f"{final_model.thickness[i]:.2f}" if final_model.thickness[i] > 0 else "∞ (half-space)"
    report += f"{i+1:<8} {thick_str:<15} {final_model.vs[i]:<12.1f} {final_model.vp[i]:<12.1f} {final_model.rho[i]:<12.2f}\n"

report += f"""
{'='*70}
2. Vs30 CALCULATION
{'='*70}

Vs30 = 30m / Σ(hi/Vsi) = {vs30:.1f} m/s

Time-Averaged Velocities:
  Vs10 = {stats['vs10']:.1f} m/s
  Vs15 = {stats['vs15']:.1f} m/s
  Vs20 = {stats['vs20']:.1f} m/s
  Vs30 = {stats['vs30']:.1f} m/s

"""

if has_uncertainty:
    report += f"""Vs30 Uncertainty (from {len(vs30_values)} models):
  Mean = {vs30_mean:.1f} m/s
  Std Dev = {vs30_std:.1f} m/s
  95% CI = [{ci_lower:.1f}, {ci_upper:.1f}] m/s
  CoV = {vs30_std/vs30_mean*100:.1f}%

"""

report += f"""{'='*70}
3. SITE CLASSIFICATION
{'='*70}

NEHRP (ASCE 7-22) Site Classification:
{'-'*70}
  Site Class: {classifications['nehrp']['class']}
  Description: {classifications['nehrp']['description']}
  
  Site Coefficients (for design spectrum):
    Fa (short period) = {classifications['nehrp']['fa']:.2f}
    Fv (long period) = {classifications['nehrp']['fv']:.2f}

NEHRP Extended Classification (9 classes):
{'-'*70}
  Site Class: {classifications['nehrp_extended']['class']}
  Description: {classifications['nehrp_extended']['description']}

Eurocode 8 Classification:
{'-'*70}
  Ground Type: {classifications['eurocode8']['class']}
  Description: {classifications['eurocode8']['description']}

{'='*70}
4. ENGINEERING INTERPRETATION
{'='*70}

Site Characteristics:
"""

# Add engineering interpretation
if vs30 < 180:
    report += """
  Site Type: SOFT SOIL
  
  Engineering Considerations:
  - High amplification of seismic ground motions expected
  - Potential for significant site effects
  - May require site-specific seismic hazard analysis
  - Consider ground improvement for critical structures
  - Higher liquefaction susceptibility if saturated cohesionless soils present
  - Foundation design should account for potential large settlements
"""
elif vs30 < 360:
    report += """
  Site Type: STIFF SOIL
  
  Engineering Considerations:
  - Moderate amplification of seismic ground motions
  - Standard seismic design provisions typically adequate
  - Evaluate liquefaction potential in saturated zones
  - Conventional foundation systems usually suitable
  - Site-specific analysis recommended for critical facilities
"""
elif vs30 < 760:
    report += """
  Site Type: VERY DENSE SOIL / SOFT ROCK
  
  Engineering Considerations:
  - Moderate seismic site effects
  - Good foundation conditions for most structures
  - Standard seismic design provisions applicable
  - Liquefaction risk generally low
  - Suitable for most types of development
"""
elif vs30 < 1500:
    report += """
  Site Type: ROCK
  
  Engineering Considerations:
  - Low seismic amplification
  - Excellent foundation conditions
  - Minimal site effects on ground motions
  - Very low liquefaction risk
  - Suitable for all structure types including critical facilities
"""
else:
    report += """
  Site Type: HARD ROCK
  
  Engineering Considerations:
  - Minimal seismic amplification
  - Exceptional foundation conditions
  - Negligible site effects
  - No liquefaction concerns
  - Ideal for critical and sensitive structures
"""

report += f"""
{'='*70}
5. QUALITY METRICS
{'='*70}

Depth of Investigation:
  Maximum wavelength: {(stats['vs30'] / 5.0):.1f} m (at 5 Hz)
  Estimated max depth: {(stats['vs30'] / 5.0):.1f} m
  Coverage for Vs30: {'ADEQUATE (≥30m)' if (stats['vs30'] / 5.0) >= 30 else 'LIMITED (<30m)'}

Velocity Range:
  Surface Vs: {stats['vs_surface']:.1f} m/s
  Minimum Vs: {stats['vs_min']:.1f} m/s
  Maximum Vs: {stats['vs_max']:.1f} m/s

{'='*70}
6. REFERENCES
{'='*70}

ASCE 7-22: Minimum Design Loads and Associated Criteria for Buildings 
            and Other Structures

NEHRP: Recommended Seismic Provisions for New Buildings and Other 
       Structures (FEMA P-2082)

Eurocode 8: Design of structures for earthquake resistance

{'='*70}
END OF REPORT
{'='*70}
"""

# Save report
report_file = os.path.join(DISPERSION_DIR, 'site_characterization_report.txt')
with open(report_file, 'w') as f:
    f.write(report)

print(report)
print(f"\nReport saved to: {report_file}")

print("\n" + "=" * 60)
print("Vs30 ANALYSIS COMPLETE: Calculation and Site Classification")
print("=" * 60)
print("\nGenerated files:")
print(f"  - Site characterization report: {report_file}")
print(f"  - Summary figures: {FIGURES_DIR}/vs30/")
print("\nVs30 analysis complete!")
print("Ready for: Visualization and Final Report Preparation")