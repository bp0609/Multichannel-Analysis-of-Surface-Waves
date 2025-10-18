# notebooks/01_data_exploration.ipynb
# Or create as: code/data_loading/explore_data.py

"""
Phase 3.1: Initial Data Inspection
Load and visualize Geophydog synthetic MASW data
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream
import os
import glob
import sys

# Add project root to path
sys.path.append('..')
from config import RAW_DATA_DIR, FIGURES_DIR, DISTANCE_CORRECTION_FACTOR

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================
# 1. LOAD ALL SAC FILES
# ============================================

def load_masw_data(data_dir):
    """Load all SAC files from directory"""
    sac_files = sorted(glob.glob(os.path.join(data_dir, "*.sac")))
    
    if not sac_files:
        raise FileNotFoundError(f"No SAC files found in {data_dir}")
    
    print(f"Found {len(sac_files)} SAC files")
    print(f"First file: {os.path.basename(sac_files[0])}")
    print(f"Last file: {os.path.basename(sac_files[-1])}")
    
    # Load all traces into a Stream
    stream = Stream()
    for sac_file in sac_files:
        st = read(sac_file)
        stream += st
    
    return stream, sac_files

# Load data
print("=" * 60)
print("LOADING GEOPHYDOG MASW DATA")
print("=" * 60)
stream, sac_files = load_masw_data(RAW_DATA_DIR)
print(f"\nLoaded {len(stream)} traces successfully\n")

# ============================================
# 2. EXTRACT DETAILED TRACE INFORMATION
# ============================================

def analyze_trace_info(stream):
    """Extract and display comprehensive trace information"""
    
    info = {
        'n_traces': len(stream),
        'sampling_rates': [],
        'n_samples': [],
        'durations': [],
        'distances': [],
        'stations': [],
        'start_times': [],
        'end_times': []
    }
    
    print("=" * 60)
    print("TRACE INFORMATION")
    print("=" * 60)
    
    for i, tr in enumerate(stream):
        # Basic info
        info['sampling_rates'].append(tr.stats.sampling_rate)
        info['n_samples'].append(tr.stats.npts)
        info['durations'].append(tr.stats.npts / tr.stats.sampling_rate)
        info['stations'].append(tr.stats.station)
        info['start_times'].append(tr.stats.starttime)
        info['end_times'].append(tr.stats.endtime)
        
        # Distance info (if available in SAC header)
        # Apply correction factor to convert from km (stored as m) to actual m
        if hasattr(tr.stats.sac, 'dist'):
            info['distances'].append(tr.stats.sac.dist * DISTANCE_CORRECTION_FACTOR)
        elif hasattr(tr.stats, 'distance'):
            info['distances'].append(tr.stats.distance * DISTANCE_CORRECTION_FACTOR)
        
        # Print first trace details
        if i == 0:
            print(f"\nFirst Trace Details (Trace {i}):")
            print(f"  Station: {tr.stats.station}")
            print(f"  Channel: {tr.stats.channel}")
            print(f"  Sampling rate: {tr.stats.sampling_rate} Hz")
            print(f"  Delta (sample interval): {tr.stats.delta} s")
            print(f"  Number of samples: {tr.stats.npts}")
            print(f"  Duration: {tr.stats.npts / tr.stats.sampling_rate:.3f} s")
            print(f"  Start time: {tr.stats.starttime}")
            print(f"  End time: {tr.stats.endtime}")
            
            # Check SAC header fields
            if hasattr(tr.stats, 'sac'):
                print(f"\n  SAC Header Fields:")
                if hasattr(tr.stats.sac, 'dist'):
                    print(f"    Distance: {tr.stats.sac.dist} m")
                if hasattr(tr.stats.sac, 'az'):
                    print(f"    Azimuth: {tr.stats.sac.az}°")
                if hasattr(tr.stats.sac, 'baz'):
                    print(f"    Back-azimuth: {tr.stats.sac.baz}°")
    
    # Summary statistics
    print(f"\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total number of traces: {info['n_traces']}")
    print(f"Sampling rate: {info['sampling_rates'][0]} Hz (all traces)")
    print(f"Nyquist frequency: {info['sampling_rates'][0]/2:.1f} Hz")
    print(f"Number of samples per trace: {info['n_samples'][0]} (all traces)")
    print(f"Record duration: {info['durations'][0]:.3f} seconds")
    
    if info['distances']:
        print(f"\nReceiver Array Configuration:")
        print(f"  Minimum offset (source to nearest receiver): {min(info['distances']):.1f} m")
        print(f"  Maximum offset: {max(info['distances']):.1f} m")
        print(f"  Array length: {max(info['distances']) - min(info['distances']):.1f} m")
        
        # Calculate spacing
        sorted_dist = sorted(info['distances'])
        spacings = np.diff(sorted_dist)
        print(f"  Receiver spacing: {np.median(spacings):.1f} m (median)")
        if len(set(spacings.round(2))) > 1:
            print(f"  Warning: Spacing varies from {min(spacings):.1f} to {max(spacings):.1f} m")
    
    return info, stream

# Analyze the data
info, stream = analyze_trace_info(stream)

# ============================================
# 3. CHECK DATA QUALITY
# ============================================

def check_data_quality(stream):
    """Check for common data quality issues"""
    
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    
    issues = []
    
    # Check 1: Missing traces
    if len(stream) < 12:
        issues.append(f"Warning: Only {len(stream)} traces (expected 24+)")
        print(f"⚠ Only {len(stream)} traces found")
    else:
        print(f"✓ Number of traces: {len(stream)}")
    
    # Check 2: Gaps or overlaps
    gaps = stream.get_gaps()
    if gaps:
        issues.append(f"Warning: {len(gaps)} gaps found in data")
        print(f"⚠ Found {len(gaps)} gaps")
    else:
        print("✓ No gaps in data")
    
    # Check 3: Consistent sampling rates
    sampling_rates = [tr.stats.sampling_rate for tr in stream]
    if len(set(sampling_rates)) > 1:
        issues.append("Warning: Inconsistent sampling rates")
        print(f"⚠ Inconsistent sampling rates: {set(sampling_rates)}")
    else:
        print(f"✓ Consistent sampling rate: {sampling_rates[0]} Hz")
    
    # Check 4: Check for NaN or Inf values
    nan_traces = []
    for i, tr in enumerate(stream):
        if np.any(np.isnan(tr.data)) or np.any(np.isinf(tr.data)):
            nan_traces.append(i)
    
    if nan_traces:
        issues.append(f"Warning: NaN/Inf values in traces {nan_traces}")
        print(f"⚠ NaN/Inf values in {len(nan_traces)} traces")
    else:
        print("✓ No NaN or Inf values")
    
    # Check 5: Check amplitude range
    print(f"\nAmplitude Statistics:")
    for i, tr in enumerate(stream):
        amp_min = tr.data.min()
        amp_max = tr.data.max()
        amp_mean = tr.data.mean()
        amp_std = tr.data.std()
        
        if i == 0:  # Print first trace as example
            print(f"  Trace {i}: min={amp_min:.2e}, max={amp_max:.2e}")
            print(f"           mean={amp_mean:.2e}, std={amp_std:.2e}")
    
    # Check 6: Zero traces
    zero_traces = []
    for i, tr in enumerate(stream):
        if np.all(tr.data == 0):
            zero_traces.append(i)
    
    if zero_traces:
        issues.append(f"Warning: {len(zero_traces)} zero traces found")
        print(f"⚠ {len(zero_traces)} traces are all zeros")
    else:
        print("✓ No zero traces")
    
    if not issues:
        print(f"\n✓ ALL QUALITY CHECKS PASSED")
    else:
        print(f"\n⚠ Found {len(issues)} quality issues")
    
    return issues

# Run quality checks
quality_issues = check_data_quality(stream)

# ============================================
# 4. VISUALIZE RAW SEISMOGRAMS (SHOT GATHER)
# ============================================

def plot_shot_gather(stream, distances=None, save_path=None):
    """
    Create shot gather plot showing all traces
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream containing all traces
    distances : array-like, optional
        Distances for each trace (m)
    save_path : str, optional
        Path to save figure
    """
    
    # Get distances if not provided
    if distances is None:
        distances = []
        for tr in stream:
            if hasattr(tr.stats.sac, 'dist'):
                distances.append(tr.stats.sac.dist * DISTANCE_CORRECTION_FACTOR)
            else:
                distances.append(0)
    
    # Ensure distances is a numpy array
    distances = np.array(distances)
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Time array
    tr = stream[0]
    time = np.arange(tr.stats.npts) * tr.stats.delta
    
    # Plot each trace with offset
    scale_factor = np.max([np.abs(tr.data).max() for tr in stream])
    
    for i, idx in enumerate(sort_idx):
        tr = stream[idx]
        dist = distances[idx]
        
        # Normalize and plot
        normalized = tr.data / scale_factor
        ax.plot(time, normalized + i, 'k-', linewidth=0.5)
        
        # Fill positive amplitudes
        ax.fill_between(time, i, normalized + i, 
                        where=(normalized > 0),
                        color='black', alpha=0.3)
    
    # Format plot
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trace Number (Distance from Source)', fontsize=12, fontweight='bold')
    ax.set_title('MASW Shot Gather - Raw Seismograms\n' + 
                 f'{len(stream)} traces, {tr.stats.sampling_rate} Hz sampling rate',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add distance labels on right
    sorted_distances = distances[sort_idx]
    yticks = np.arange(len(stream))
    ax.set_yticks(yticks[::2])  # Every other trace to avoid crowding
    ax.set_yticklabels([f'{sorted_distances[i]:.1f} m' for i in yticks[::2]])
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

# Create shot gather plot
print("\n" + "=" * 60)
print("GENERATING SHOT GATHER PLOT")
print("=" * 60)

fig_shotgather = plot_shot_gather(
    stream, 
    distances=info['distances'],
    save_path=os.path.join(FIGURES_DIR, 'shot_gather_raw.png')
)

# ============================================
# 5. PLOT INDIVIDUAL TRACES WITH DETAILS
# ============================================

def plot_individual_traces(stream, n_traces=6, save_path=None):
    """
    Plot individual traces to examine waveforms in detail
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream containing traces
    n_traces : int
        Number of traces to plot
    save_path : str, optional
        Path to save figure
    """
    
    # Select evenly spaced traces
    indices = np.linspace(0, len(stream)-1, n_traces, dtype=int)
    
    fig, axes = plt.subplots(n_traces, 1, figsize=(12, 10))
    
    for i, idx in enumerate(indices):
        tr = stream[idx]
        time = np.arange(tr.stats.npts) * tr.stats.delta
        
        # Get distance (apply correction factor)
        dist = 0
        if hasattr(tr.stats.sac, 'dist'):
            dist = tr.stats.sac.dist * DISTANCE_CORRECTION_FACTOR
        
        # Plot
        axes[i].plot(time, tr.data, 'k-', linewidth=0.8)
        axes[i].set_ylabel(f'Trace {idx}\n{dist:.1f} m', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if i < n_traces - 1:
            axes[i].set_xticklabels([])
        else:
            axes[i].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    
    fig.suptitle('Individual Seismogram Traces\nDetailed Waveform View', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

# Plot individual traces
fig_individual = plot_individual_traces(
    stream,
    n_traces=6,
    save_path=os.path.join(FIGURES_DIR, 'individual_traces.png')
)

# ============================================
# 6. FREQUENCY ANALYSIS
# ============================================

def plot_frequency_content(stream, save_path=None):
    """
    Analyze and plot frequency content of seismograms
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream containing traces
    save_path : str, optional
        Path to save figure
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Select a few traces to analyze
    indices = [0, len(stream)//2, len(stream)-1]
    colors = ['blue', 'green', 'red']
    labels = ['Near offset', 'Mid offset', 'Far offset']
    
    for idx, color, label in zip(indices, colors, labels):
        tr = stream[idx]
        
        # Compute FFT
        fft = np.fft.rfft(tr.data)
        freqs = np.fft.rfftfreq(tr.stats.npts, tr.stats.delta)
        amplitudes = np.abs(fft)
        
        # Plot amplitude spectrum
        axes[0].plot(freqs, amplitudes, color=color, label=label, alpha=0.7, linewidth=1.5)
        
        # Plot power spectrum (log scale)
        power = amplitudes**2
        axes[1].semilogy(freqs, power, color=color, label=label, alpha=0.7, linewidth=1.5)
    
    # Format amplitude spectrum
    axes[0].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Amplitude Spectrum', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 100)  # Focus on 0-100 Hz
    
    # Format power spectrum
    axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1].set_ylabel('Power (log scale)', fontsize=11)
    axes[1].set_title('Power Spectrum (Log Scale)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].set_xlim(0, 100)
    
    # Add typical MASW frequency band
    for ax in axes:
        ax.axvspan(5, 50, alpha=0.1, color='orange', label='Typical MASW band (5-50 Hz)')
    
    fig.suptitle('Frequency Content Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

# Analyze frequency content
print("\n" + "=" * 60)
print("ANALYZING FREQUENCY CONTENT")
print("=" * 60)

fig_freq = plot_frequency_content(
    stream,
    save_path=os.path.join(FIGURES_DIR, 'frequency_content.png')
)

# ============================================
# 7. ACQUISITION GEOMETRY VERIFICATION
# ============================================

def plot_acquisition_geometry(stream, save_path=None):
    """
    Visualize acquisition geometry
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream containing traces
    save_path : str, optional
        Path to save figure
    """
    
    # Extract distances (apply correction factor)
    distances = []
    for tr in stream:
        if hasattr(tr.stats.sac, 'dist'):
            distances.append(tr.stats.sac.dist * DISTANCE_CORRECTION_FACTOR)
        else:
            distances.append(0)
    
    distances = np.array(distances)
    sorted_distances = np.sort(distances)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Schematic of geometry
    ax1 = axes[0]
    
    # Draw source
    ax1.scatter([0], [0], s=300, c='red', marker='*', 
                label='Source', zorder=3, edgecolors='black', linewidth=2)
    
    # Draw receivers
    ax1.scatter(sorted_distances, np.zeros_like(sorted_distances), 
                s=150, c='blue', marker='v', label='Receivers', 
                zorder=2, edgecolors='black', linewidth=1)
    
    # Draw surface
    ax1.plot([sorted_distances[0]-5, sorted_distances[-1]+5], [0, 0], 
             'k-', linewidth=2, label='Surface')
    
    # Format
    ax1.set_xlabel('Distance from Source (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('', fontsize=11)
    ax1.set_title('MASW Acquisition Geometry - Plan View', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2, 2)
    ax1.set_yticks([])
    
    # Add annotations
    ax1.annotate(f'X1 = {sorted_distances[0]:.1f} m\n(Source offset)', 
                xy=(sorted_distances[0]/2, 0.5), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    if len(sorted_distances) > 1:
        spacing = np.median(np.diff(sorted_distances))
        ax1.annotate(f'dx = {spacing:.1f} m\n(Receiver spacing)', 
                    xy=(sorted_distances[-2], -0.5), fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Plot 2: Distance distribution
    ax2 = axes[1]
    
    ax2.plot(range(len(sorted_distances)), sorted_distances, 'o-', 
             color='darkblue', markersize=8, linewidth=2)
    ax2.set_xlabel('Receiver Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Distance from Source (m)', fontsize=11, fontweight='bold')
    ax2.set_title('Receiver Positions', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Array Statistics:\n'
    stats_text += f'Minimum offset: {sorted_distances[0]:.1f} m\n'
    stats_text += f'Maximum offset: {sorted_distances[-1]:.1f} m\n'
    stats_text += f'Array length: {sorted_distances[-1] - sorted_distances[0]:.1f} m\n'
    stats_text += f'Number of receivers: {len(sorted_distances)}'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig

# Plot geometry
print("\n" + "=" * 60)
print("VISUALIZING ACQUISITION GEOMETRY")
print("=" * 60)

fig_geometry = plot_acquisition_geometry(
    stream,
    save_path=os.path.join(FIGURES_DIR, 'acquisition_geometry.png')
)

print("\n" + "=" * 60)
print("PHASE 3.1 COMPLETE: Initial Data Inspection")
print("=" * 60)
print("\nGenerated figures:")
print("  1. shot_gather_raw.png")
print("  2. individual_traces.png")
print("  3. frequency_content.png")
print("  4. acquisition_geometry.png")
print("\nProceed to Phase 3.2: Preprocessing")