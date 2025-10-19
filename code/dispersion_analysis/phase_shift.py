# code/dispersion_analysis/phase_shift.py

"""
Phase Shift Method for MASW Dispersion Curve Extraction
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm

def phase_shift_transform(data, dt, offsets, frequencies, velocities, 
                         wave_type='cylindrical'):
    """
    Phase shift method to extract dispersion curves
    
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
    wave_type : str
        'plane' for plane wave assumption
        'cylindrical' for point source (more accurate for MASW)
    
    Returns:
    --------
    dispersion_image : ndarray
        2D array (frequencies x velocities) with normalized energy
    """
    
    n_traces, n_samples = data.shape
    n_freqs = len(frequencies)
    n_vels = len(velocities)
    
    # Initialize dispersion image
    dispersion_image = np.zeros((n_freqs, n_vels))
    
    # Compute FFT of all traces
    fft_data = np.fft.rfft(data, axis=1)
    freq_axis = np.fft.rfftfreq(n_samples, dt)
    
    print("Computing phase shift transform...")
    
    # Loop over each trial frequency
    for i, f in enumerate(tqdm(frequencies, desc="Frequencies")):
        # Find nearest frequency bin
        f_idx = np.argmin(np.abs(freq_axis - f))
        
        # Extract complex amplitudes at this frequency
        U = fft_data[:, f_idx]
        
        # Loop over each trial velocity
        for j, c in enumerate(velocities):
            # Calculate phase shifts for each receiver
            if wave_type == 'plane':
                # Plane wave assumption
                phase_shifts = 2 * np.pi * f * offsets / c
            else:  # cylindrical
                # Point source (cylindrical wave) - more accurate
                # Include amplitude correction for geometric spreading
                amp_correction = 1.0 / np.sqrt(offsets)
                phase_shifts = 2 * np.pi * f * offsets / c
                U_corrected = U * amp_correction
                U = U_corrected
            
            # Apply phase shifts and sum coherently
            shifted = U * np.exp(-1j * phase_shifts)
            stacked = np.sum(shifted)
            
            # Store normalized energy (power)
            dispersion_image[i, j] = np.abs(stacked) ** 2
    
    # Normalize each frequency slice
    for i in range(n_freqs):
        max_val = dispersion_image[i, :].max()
        if max_val > 0:
            dispersion_image[i, :] /= max_val
    
    print("Phase shift transform complete!")
    
    return dispersion_image


def fk_transform(data, dt, dx, freq_range=None, vel_range=None):
    """
    Frequency-wavenumber (f-k) transform method
    
    Parameters:
    -----------
    data : ndarray
        Seismic data array (n_traces x n_samples)
    dt : float
        Time sampling interval (s)
    dx : float
        Spatial sampling interval (m)
    freq_range : tuple, optional
        (fmin, fmax) in Hz
    vel_range : tuple, optional
        (vmin, vmax) in m/s
    
    Returns:
    --------
    dispersion_image : ndarray
        2D array with dispersion image
    frequencies : ndarray
        Frequency array
    velocities : ndarray
        Velocity array
    """
    
    n_traces, n_samples = data.shape
    
    # 2D FFT
    print("Computing 2D FFT (f-k transform)...")
    fk_spectrum = np.fft.fft2(data)
    fk_spectrum = np.fft.fftshift(fk_spectrum, axes=0)
    
    # Frequency and wavenumber axes
    freqs = np.fft.fftfreq(n_samples, dt)
    freqs = np.fft.fftshift(freqs)
    
    wavenumbers = np.fft.fftfreq(n_traces, dx)
    wavenumbers = np.fft.fftshift(wavenumbers)
    
    # Take absolute value and focus on positive frequencies
    fk_amplitude = np.abs(fk_spectrum)
    
    # Select positive frequencies only
    pos_freq_idx = freqs > 0
    freqs_pos = freqs[pos_freq_idx]
    fk_amplitude_pos = fk_amplitude[:, pos_freq_idx]
    
    # Convert wavenumber to velocity: v = 2*pi*f/k
    # Create meshgrid
    K, F = np.meshgrid(wavenumbers, freqs_pos)
    
    # Avoid division by zero
    K_safe = K.copy()
    K_safe[K_safe == 0] = 1e-10
    
    # Calculate velocities
    V = 2 * np.pi * F / K_safe
    
    # Filter by velocity range if specified
    if vel_range is not None:
        vmin, vmax = vel_range
        valid = (V >= vmin) & (V <= vmax)
        fk_amplitude_pos[~valid.T] = 0
    
    # Filter by frequency range if specified
    if freq_range is not None:
        fmin, fmax = freq_range
        valid_f = (freqs_pos >= fmin) & (freqs_pos <= fmax)
        fk_amplitude_pos[:, ~valid_f] = 0
    
    # Interpolate to regular velocity grid
    if vel_range is not None:
        vmin, vmax = vel_range
    else:
        vmin, vmax = 100, 1000
    
    velocities = np.linspace(vmin, vmax, 200)
    
    # Create dispersion image by resampling
    from scipy.interpolate import griddata
    
    dispersion_image = np.zeros((len(freqs_pos), len(velocities)))
    
    for i in range(len(freqs_pos)):
        # Extract velocity and amplitude for this frequency
        v_slice = V[:, i]
        amp_slice = fk_amplitude_pos[:, i]
        
        # Interpolate to regular velocity grid
        valid = np.isfinite(v_slice) & (v_slice > 0)
        if np.sum(valid) > 0:
            dispersion_image[i, :] = np.interp(velocities, 
                                               v_slice[valid], 
                                               amp_slice[valid],
                                               left=0, right=0)
    
    # Normalize
    for i in range(len(freqs_pos)):
        max_val = dispersion_image[i, :].max()
        if max_val > 0:
            dispersion_image[i, :] /= max_val
    
    print("F-K transform complete!")
    
    return dispersion_image, freqs_pos, velocities


def slant_stack_transform(data, dt, offsets, frequencies, velocities):
    """
    Slant-stack (tau-p) transform method
    
    Parameters:
    -----------
    data : ndarray
        Seismic data array (n_traces x n_samples)
    dt : float
        Time sampling interval (s)
    offsets : ndarray
        Source-receiver distances (m)
    frequencies : ndarray
        Frequency array (Hz)
    velocities : ndarray
        Phase velocity array (m/s)
    
    Returns:
    --------
    dispersion_image : ndarray
        2D dispersion image
    """
    
    n_traces, n_samples = data.shape
    time = np.arange(n_samples) * dt
    
    n_freqs = len(frequencies)
    n_vels = len(velocities)
    
    dispersion_image = np.zeros((n_freqs, n_vels))
    
    print("Computing slant-stack transform...")
    
    for j, c in enumerate(tqdm(velocities, desc="Velocities")):
        # Slowness (s/m)
        p = 1.0 / c
        
        # Stack along moveout curve
        stacked_trace = np.zeros(n_samples)
        
        for i, offset in enumerate(offsets):
            # Time shift for this offset
            time_shift = offset * p
            
            # Shift trace
            if time_shift < time[-1]:
                shifted = np.interp(time - time_shift, time, data[i, :], 
                                   left=0, right=0)
                stacked_trace += shifted
        
        # Average
        stacked_trace /= n_traces
        
        # Compute amplitude spectrum of stacked trace
        fft_stack = np.fft.rfft(stacked_trace)
        freq_axis = np.fft.rfftfreq(n_samples, dt)
        amp_spectrum = np.abs(fft_stack)
        
        # Extract amplitudes at desired frequencies
        for i, f in enumerate(frequencies):
            f_idx = np.argmin(np.abs(freq_axis - f))
            dispersion_image[i, j] = amp_spectrum[f_idx]
    
    # Normalize each frequency
    for i in range(n_freqs):
        max_val = dispersion_image[i, :].max()
        if max_val > 0:
            dispersion_image[i, :] /= max_val
    
    print("Slant-stack transform complete!")
    
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
    
    # Plot dispersion image
    extent = [velocities.min(), velocities.max(), 
              frequencies.min(), frequencies.max()]
    
    im = ax.imshow(dispersion_image, aspect=aspect, origin='lower',
                   extent=extent, cmap=cmap, interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Energy', fontsize=11, fontweight='bold')
    
    # Plot picked curve if provided
    if picked_curve is not None:
        ax.plot(picked_curve, frequencies, 'w-', linewidth=3, 
                label='Fundamental Mode', alpha=0.9)
        ax.plot(picked_curve, frequencies, 'k--', linewidth=1.5, alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
    
    # Labels and formatting
    ax.set_xlabel('Phase Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
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
        
        # Estimate uncertainty from peak width
        # Find half-maximum points
        half_max = energy[max_idx] * 0.5
        above_half = energy > half_max
        
        if np.sum(above_half) > 1:
            # Width at half maximum
            indices = np.where(above_half)[0]
            width = velocities[indices[-1]] - velocities[indices[0]]
            uncertainties[i] = width / 2.355  # Convert FWHM to std dev
        else:
            # Use velocity resolution
            uncertainties[i] = (velocities[1] - velocities[0]) * 2
    
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