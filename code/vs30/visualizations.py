# code/vs30/visualizations.py

"""
Visualization functions for Vs30 and site classification
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vs30.site_classification import NEHRPClassification, classify_site_all_systems
from vs30.calculate_vs30 import calculate_vs30, calculate_vs_statistics

def plot_vs_profile_with_vs30(model, save_path=None):
    """
    Plot Vs profile with Vs30 highlighted
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    save_path : str, optional
        Path to save figure
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get profile
    depths, vs_profile = model.get_depth_array(dz=0.5)
    
    # Calculate Vs30
    vs30 = calculate_vs30(model)
    
    # Calculate average Vs in top 30m for comparison
    mask_30m = depths <= 30
    avg_vs_30m = np.mean(vs_profile[mask_30m])
    
    # Plot Vs profile
    ax.plot(vs_profile, depths, 'b-', linewidth=3, label='Vs Profile')
    
    # Mark layer boundaries
    cumulative_depth = 0
    for i in range(model.n_layers - 1):
        cumulative_depth += model.thickness[i]
        ax.axhline(cumulative_depth, color='gray', linestyle='--', 
                   alpha=0.5, linewidth=1)
        ax.text(model.vs[i], cumulative_depth - 1, 
                f'  Layer {i+1}: {model.vs[i]:.0f} m/s', 
                fontsize=9, verticalalignment='top')
    
    # Highlight 30m depth
    ax.axhline(30, color='red', linestyle='-', linewidth=2.5, 
               label='30m Depth', alpha=0.7, zorder=10)
    
    # Draw Vs30 line
    ax.axvline(vs30, color='green', linestyle='--', linewidth=2.5,
               label=f'Vs30 = {vs30:.1f} m/s', alpha=0.8, zorder=9)
    
    # Shade the top 30m region
    ax.axhspan(0, 30, alpha=0.1, color='yellow', zorder=1)
    
    # Add text annotations
    ax.text(0.02, 0.98, f'Vs30 = {vs30:.1f} m/s\nArithmetic Mean = {avg_vs_30m:.1f} m/s',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Get site classification
    classifications = classify_site_all_systems(vs30)
    nehrp = classifications['nehrp']
    
    ax.text(0.02, 0.75, 
            f"NEHRP Site Class: {nehrp['class']}\n{nehrp['description']}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Vs (m/s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=13, fontweight='bold')
    ax.set_title('Shear Wave Velocity Profile with Vs30', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(50, 0)
    ax.set_xlim(0, max(model.vs) * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_nehrp_classification_chart(vs30, save_path=None):
    """
    Create NEHRP site classification chart with current site marked
    
    Parameters:
    -----------
    vs30 : float
        Vs30 value (m/s)
    save_path : str, optional
        Path to save figure
    """
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # NEHRP boundaries
    boundaries = [0, 180, 360, 760, 1500, 2500]
    labels = ['E', 'D', 'C', 'B', 'A']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    descriptions = [
        'Soft Soil',
        'Stiff Soil',
        'Very Dense Soil\nand Soft Rock',
        'Rock',
        'Hard Rock'
    ]
    
    # Draw bars
    y_pos = 0.5
    bar_height = 0.8
    
    for i in range(len(labels)):
        width = boundaries[i+1] - boundaries[i]
        rect = Rectangle((boundaries[i], y_pos - bar_height/2), width, bar_height,
                        facecolor=colors[i], edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Add class label
        center_x = (boundaries[i] + boundaries[i+1]) / 2
        ax.text(center_x, y_pos, f'Class {labels[i]}\n{descriptions[i]}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', bbox=dict(boxstyle='round', 
                facecolor='black', alpha=0.5))
    
    # Mark current site
    arrow_y = y_pos + bar_height/2 + 0.3
    ax.annotate(f'Your Site\nVs30 = {vs30:.0f} m/s',
                xy=(vs30, y_pos + bar_height/2), 
                xytext=(vs30, arrow_y),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    
    # Vertical line at site
    ax.axvline(vs30, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Vs30 (m/s)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    ax.set_title('NEHRP Site Classification (ASCE 7-22)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add boundary values
    for boundary in boundaries[1:-1]:
        ax.text(boundary, -0.05, f'{boundary}', ha='center', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_vs_statistics(model, save_path=None):
    """
    Plot various Vs statistics (Vs10, Vs15, Vs20, Vs30, etc.)
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    save_path : str, optional
        Path to save figure
    """
    
    stats = calculate_vs_statistics(model)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Vs profile with multiple depth markers
    ax = axes[0]
    
    depths, vs_profile = model.get_depth_array(dz=0.5)
    ax.plot(vs_profile, depths, 'b-', linewidth=3, label='Vs Profile')
    
    # Mark different depth intervals
    depth_markers = [
        (10, stats['vs10'], 'Vs10', 'green'),
        (15, stats['vs15'], 'Vs15', 'orange'),
        (20, stats['vs20'], 'Vs20', 'purple'),
        (30, stats['vs30'], 'Vs30', 'red')
    ]
    
    for depth, vs_value, label, color in depth_markers:
        ax.axhline(depth, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(vs_value, color=color, linestyle=':', linewidth=2, alpha=0.7)
        ax.plot(vs_value, depth, 'o', color=color, markersize=12, 
                label=f'{label} = {vs_value:.0f} m/s', zorder=10)
    
    ax.set_xlabel('Vs (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('Vs Profile with Multiple Depth Averages', 
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(50, 0)
    
    # Plot 2: Bar chart of VsZ values
    ax = axes[1]
    
    depths_z = [10, 15, 20, 30]
    vs_z_values = [stats['vs10'], stats['vs15'], stats['vs20'], stats['vs30']]
    colors_bar = ['green', 'orange', 'purple', 'red']
    
    bars = ax.bar(depths_z, vs_z_values, width=3, color=colors_bar, 
                  edgecolor='black', linewidth=2, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, vs_z_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f} m/s', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time-Averaged Vs (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Time-Averaged Shear Wave Velocities\nat Different Depths', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticks(depths_z)
    ax.set_xticklabels([f'Vs{d}' for d in depths_z])
    
    fig.suptitle('Shear Wave Velocity Statistics', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_vs30_uncertainty(vs30_values, save_path=None):
    """
    Plot Vs30 uncertainty distribution
    
    Parameters:
    -----------
    vs30_values : array-like
        Array of Vs30 values from uncertainty analysis
    save_path : str, optional
        Path to save figure
    """
    
    vs30_mean = np.mean(vs30_values)
    vs30_std = np.std(vs30_values)
    vs30_min = np.min(vs30_values)
    vs30_max = np.max(vs30_values)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Histogram
    ax = axes[0]
    
    n, bins, patches = ax.hist(vs30_values, bins=30, density=True,
                               alpha=0.7, color='blue', edgecolor='black')
    
    # Add normal distribution overlay
    from scipy.stats import norm
    x = np.linspace(vs30_min, vs30_max, 100)
    ax.plot(x, norm.pdf(x, vs30_mean, vs30_std), 'r-', linewidth=3,
            label=f'Normal fit\n(μ={vs30_mean:.1f}, σ={vs30_std:.1f})')
    
    # Mark mean and std dev
    ax.axvline(vs30_mean, color='green', linestyle='--', linewidth=2.5,
               label=f'Mean = {vs30_mean:.1f} m/s')
    ax.axvline(vs30_mean - vs30_std, color='orange', linestyle=':', linewidth=2)
    ax.axvline(vs30_mean + vs30_std, color='orange', linestyle=':', linewidth=2,
               label=f'±1σ = {vs30_std:.1f} m/s')
    
    ax.set_xlabel('Vs30 (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Vs30 Probability Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Classification probabilities
    ax = axes[1]
    
    # Count how many fall in each NEHRP class
    classes = ['E', 'D', 'C', 'B', 'A']
    boundaries = [0, 180, 360, 760, 1500, np.inf]
    counts = []
    
    for i in range(len(classes)):
        count = np.sum((vs30_values >= boundaries[i]) & 
                      (vs30_values < boundaries[i+1]))
        counts.append(count)
    
    percentages = np.array(counts) / len(vs30_values) * 100
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bars = ax.bar(classes, percentages, color=colors, edgecolor='black',
                  linewidth=2, alpha=0.7)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        if pct > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # Determine most likely class
    most_likely_idx = np.argmax(percentages)
    most_likely_class = classes[most_likely_idx]
    
    ax.set_xlabel('NEHRP Site Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Site Classification Probability\nMost Likely: Class {most_likely_class}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(percentages) * 1.2)
    
    fig.suptitle(f'Vs30 Uncertainty Analysis ({len(vs30_values)} models)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig