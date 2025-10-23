# code/dispersion_analysis/phase_shift.py

"""
Phase Shift Method for MASW Dispersion Curve Extraction
Based on Geophydog's masw_fc_trans implementation
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm

def phase_shift_transform(data, dt, offsets, frequencies, velocities):
    """
    Phase shift method to extract dispersion curves (f-c transform)
    
    Based on Geophydog's masw_fc_trans implementation.
    Uses phase-only stacking (normalized FFT) for dispersion analysis.
    
    Parameters:
    -----------
    data : ndarray
        Seismic data array (n_traces x n_samples)
    dt : float
        Time sampling interval (s)
    offsets : ndarray
        Source-receiver distances (m)
    frequencies : ndarray
        Frequency array to analyze (Hz)
    velocities : ndarray
        Trial phase velocities (m/s)
    
    Returns:
    --------
    dispersion_image : ndarray
        2D array (frequencies x velocities) with normalized energy
    """
    
    m, n = data.shape
    fs = 1.0 / dt
    
    # Get frequency bounds from input
    f1 = frequencies[0]
    f2 = frequencies[-1]
    
    # Frequency array
    f = np.arange(n) * fs / (n - 1)
    
    # Get frequency indices for f1..f2
    fn1 = int(max(0, np.floor(f1 * (n-1) / fs)))
    fn2 = int(min(n - 1, np.ceil(f2 * (n-1) / fs)))
    
    w = 2.0 * np.pi * f
    
    print("Computing phase shift transform...")
    
    # FFT of each trace
    fft_d = np.zeros((m, n), dtype=np.complex128)
    for i in range(m):
        fft_d[i] = np.fft.fft(data[i])
    
    # Normalize by magnitude (phase-only stack)
    with np.errstate(divide='ignore', invalid='ignore'):
        fft_d = fft_d / np.abs(fft_d)
    fft_d[np.isnan(fft_d)] = 0.0
    
    # Initialize output
    fc = np.zeros((len(velocities), fn2 - fn1 + 1), dtype=np.float64)
    
    # Phase shift stack
    for ci, cc in enumerate(tqdm(velocities, desc="Velocities")):
        for fi in range(fn1, fn2 + 1):
            # exp(i * w / c * dist)
            phase_delays = np.exp(1j * w[fi] / cc * offsets)
            # sum across channels
            val = np.abs(np.sum(phase_delays * fft_d[:, fi]))
            fc[ci, fi - fn1] = val
    
    # Normalize for plotting
    norm = np.abs(fc).max()
    if norm > 0:
        fc = fc / norm
    
    # Interpolate to requested frequency grid
    freqs_computed = f[fn1:fn2 + 1]
    
    # Create output dispersion image matching requested dimensions
    dispersion_image = np.zeros((len(frequencies), len(velocities)))
    
    for j in range(len(velocities)):
        dispersion_image[:, j] = np.interp(frequencies, freqs_computed, fc[j, :])
    
    print("Phase shift transform complete!")
    
    return dispersion_image


def plot_dispersion_image(dispersion_image, frequencies, velocities, 
                          picked_curve=None, title="Dispersion Image",
                          save_path=None, cmap='jet', aspect='auto'):
    """
    Plot dispersion image with optional picked curve
    
    Parameters:
    -----------
    dispersion_image : ndarray
        2D dispersion image
    frequencies : ndarray
        Frequency array (Hz)
    velocities : ndarray
        Velocity array (m/s)
    picked_curve : ndarray, optional
        Picked dispersion curve velocities at each frequency
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    cmap : str
        Colormap name
    aspect : str
        Aspect ratio
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot dispersion image (exchange x and y axes)
    extent = [frequencies.min(), frequencies.max(),
              velocities.min(), velocities.max()]
    
    # Notice transpose so that v is y, f is x
    im = ax.imshow(dispersion_image.T, aspect=aspect, origin='lower',
                   extent=extent, cmap=cmap, interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Energy', fontsize=11, fontweight='bold')
    
    # Plot picked curve if provided (swap axes: f on x, v on y)
    if picked_curve is not None:
        ax.plot(frequencies, picked_curve, 'w-', linewidth=3, 
                label='Fundamental Mode', alpha=0.9)
        ax.plot(frequencies, picked_curve, 'k--', linewidth=1.5, alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
    
    # Labels and formatting (swap x and y, i.e., x is frequency, y is velocity)
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig, ax


def automatic_picking(dispersion_image, frequencies, velocities, 
                     smooth_window=3, threshold=0.5):
    """
    Automatically pick dispersion curve from dispersion image
    
    Parameters:
    -----------
    dispersion_image : ndarray
        2D dispersion image
    frequencies : ndarray
        Frequency array
    velocities : ndarray
        Velocity array
    smooth_window : int
        Window size for smoothing picked curve
    threshold : float
        Minimum normalized energy threshold (0-1)
    
    Returns:
    --------
    picked_velocities : ndarray
        Picked phase velocities at each frequency
    uncertainties : ndarray
        Uncertainty estimates (standard deviation)
    """
    
    n_freqs = len(frequencies)
    picked_velocities = np.zeros(n_freqs)
    uncertainties = np.zeros(n_freqs)
    
    for i in range(n_freqs):
        # Extract energy at this frequency
        energy = dispersion_image[i, :]
        
        # Find maximum energy
        max_idx = np.argmax(energy)
        picked_velocities[i] = velocities[max_idx]
        
        # Estimate uncertainty from peak width at 80% of maximum
        # Using 80% instead of 50% gives more realistic uncertainty for sharp peaks
        # This is more appropriate for dispersion curves with good SNR
        threshold = energy[max_idx] * 0.8
        above_threshold = energy > threshold
        
        if np.sum(above_threshold) > 1:
            # Width at 80% maximum
            indices = np.where(above_threshold)[0]
            width = velocities[indices[-1]] - velocities[indices[0]]
            # Use a fraction of the width as uncertainty (more conservative than just width/2)
            uncertainties[i] = width / 3.0
        else:
            # Use velocity resolution as minimum uncertainty
            vel_resolution = velocities[1] - velocities[0]
            uncertainties[i] = vel_resolution * 2
        
        # Cap maximum uncertainty at 10% of picked velocity or 50 m/s, whichever is smaller
        # This prevents unrealistically large uncertainties
        max_uncertainty = min(picked_velocities[i] * 0.1, 50.0)
        uncertainties[i] = min(uncertainties[i], max_uncertainty)
    
    # Smooth the picked curve
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        picked_velocities = uniform_filter1d(picked_velocities, 
                                            size=smooth_window)
    
    print(f"Automatic picking complete")
    print(f"  Velocity range: {picked_velocities.min():.1f} - {picked_velocities.max():.1f} m/s")
    print(f"  Mean uncertainty: {uncertainties.mean():.1f} m/s")
    
    return picked_velocities, uncertainties


def interactive_picking(dispersion_image, frequencies, velocities, 
                       n_modes=1, save_path=None):
    """
    Interactive manual picking of dispersion curves
    
    Parameters:
    -----------
    dispersion_image : ndarray
        2D dispersion image
    frequencies : ndarray
        Frequency array
    velocities : ndarray
        Velocity array
    n_modes : int
        Number of modes to pick
    save_path : str, optional
        Path to save picked curves
    
    Returns:
    --------
    picked_curves : list of ndarray
        List of picked curves (one per mode)
    """
    
    print("\n" + "=" * 60)
    print("INTERACTIVE DISPERSION CURVE PICKING")
    print("=" * 60)
    print("Instructions:")
    print("  - Click on the dispersion image to pick points")
    print("  - Pick from low to high frequency")
    print("  - Press 'Enter' when done with current mode")
    print("  - Close window when all modes are picked")
    print("=" * 60 + "\n")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot dispersion image
    extent = [velocities.min(), velocities.max(), 
              frequencies.min(), frequencies.max()]
    
    im = ax.imshow(dispersion_image, aspect='auto', origin='lower',
                   extent=extent, cmap='jet', interpolation='bilinear')
    
    plt.colorbar(im, ax=ax, label='Normalized Energy')
    
    ax.set_xlabel('Phase Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('Click to pick dispersion curve\nPress Enter for next mode', 
                fontsize=12, fontweight='bold')
    
    picked_curves = []
    colors = ['white', 'yellow', 'cyan']
    
    for mode in range(n_modes):
        print(f"\nPicking Mode {mode}...")
        points = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_stop=2)
        
        if len(points) > 0:
            points = np.array(points)
            # Sort by frequency
            sort_idx = np.argsort(points[:, 1])
            points = points[sort_idx]
            
            # Interpolate to regular frequency grid
            picked_vels = np.interp(frequencies, points[:, 1], points[:, 0])
            picked_curves.append(picked_vels)
            
            # Plot picked curve
            color = colors[mode % len(colors)]
            ax.plot(picked_vels, frequencies, color=color, 
                   linewidth=3, label=f'Mode {mode}', alpha=0.9)
            ax.plot(picked_vels, frequencies, 'k--', linewidth=1, alpha=0.5)
            ax.legend()
            plt.draw()
            
            print(f"  Picked {len(points)} points")
        else:
            print(f"  No points picked for mode {mode}")
    
    plt.close()
    
    # Save picked curves if requested
    if save_path and picked_curves:
        data = np.column_stack([frequencies] + picked_curves)
        header = 'Frequency(Hz) ' + ' '.join([f'Mode{i}(m/s)' for i in range(len(picked_curves))])
        np.savetxt(save_path, data, header=header, fmt='%.3f')
        print(f"\nPicked curves saved to: {save_path}")
    
    return picked_curves