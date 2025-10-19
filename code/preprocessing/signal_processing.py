# code/preprocessing/signal_processing.py

"""
Signal processing functions for MASW data preprocessing
"""

import numpy as np
from scipy import signal
from obspy import Stream
import matplotlib.pyplot as plt

def bandpass_filter(stream, freqmin=5.0, freqmax=50.0, corners=4, zerophase=True):
    """
    Apply bandpass filter to remove noise outside surface wave band
    
    Parameters:
    -----------
    stream : obspy.Stream
        Input stream
    freqmin : float
        Minimum frequency (Hz)
    freqmax : float
        Maximum frequency (Hz)
    corners : int
        Filter corners
    zerophase : bool
        If True, apply zero-phase filter
    
    Returns:
    --------
    stream_filtered : obspy.Stream
        Filtered stream
    """
    
    stream_filtered = stream.copy()
    stream_filtered.filter('bandpass', freqmin=freqmin, freqmax=freqmax,
                          corners=corners, zerophase=zerophase)
    
    print(f"Applied bandpass filter: {freqmin}-{freqmax} Hz")
    print(f"  Corners: {corners}, Zero-phase: {zerophase}")
    
    return stream_filtered


def normalize_traces(stream, method='trace'):
    """
    Normalize trace amplitudes
    
    Parameters:
    -----------
    stream : obspy.Stream
        Input stream
    method : str
        'trace' - normalize each trace to its own maximum
        'global' - normalize all traces to global maximum
        'rms' - normalize by RMS amplitude
    
    Returns:
    --------
    stream_normalized : obspy.Stream
        Normalized stream
    """
    
    stream_normalized = stream.copy()
    
    if method == 'trace':
        for tr in stream_normalized:
            max_amp = np.abs(tr.data).max()
            if max_amp > 0:
                tr.data = tr.data / max_amp
        print("Applied trace-by-trace normalization")
    
    elif method == 'global':
        global_max = max([np.abs(tr.data).max() for tr in stream_normalized])
        for tr in stream_normalized:
            tr.data = tr.data / global_max
        print(f"Applied global normalization (max = {global_max:.2e})")
    
    elif method == 'rms':
        for tr in stream_normalized:
            rms = np.sqrt(np.mean(tr.data**2))
            if rms > 0:
                tr.data = tr.data / rms
        print("Applied RMS normalization")
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return stream_normalized


def apply_muting(stream, mute_start=None, mute_end=None, distances=None, 
                 velocity_min=None, velocity_max=None):
    """
    Apply muting to remove unwanted arrivals (e.g., body waves)
    
    Parameters:
    -----------
    stream : obspy.Stream
        Input stream
    mute_start : float, optional
        Time to start muting (s)
    mute_end : float, optional
        Time to end muting (s)
    distances : array-like, optional
        Distances for velocity-based muting
    velocity_min : float, optional
        Minimum velocity for moveout muting (m/s)
    velocity_max : float, optional
        Maximum velocity for moveout muting (m/s)
    
    Returns:
    --------
    stream_muted : obspy.Stream
        Muted stream
    """
    
    stream_muted = stream.copy()
    
    if mute_start is not None and mute_end is not None:
        # Time-based muting
        for tr in stream_muted:
            n_start = int(mute_start / tr.stats.delta)
            n_end = int(mute_end / tr.stats.delta)
            tr.data[n_start:n_end] = 0
        print(f"Applied time muting: {mute_start}-{mute_end} s")
    
    elif velocity_min is not None and distances is not None:
        # Velocity-based muting (for body waves)
        for i, tr in enumerate(stream_muted):
            dist = distances[i]
            t_arrive = dist / velocity_min
            n_mute = int(t_arrive / tr.stats.delta)
            tr.data[:n_mute] = 0
        print(f"Applied velocity-based muting: v > {velocity_min} m/s")
    
    return stream_muted


def spectral_whitening(stream, smooth_window=1.0):
    """
    Apply spectral whitening to balance frequency content
    
    Parameters:
    -----------
    stream : obspy.Stream
        Input stream
    smooth_window : float
        Smoothing window in Hz
    
    Returns:
    --------
    stream_whitened : obspy.Stream
        Whitened stream
    """
    
    stream_whitened = stream.copy()
    
    for tr in stream_whitened:
        # Compute FFT
        fft = np.fft.rfft(tr.data)
        freqs = np.fft.rfftfreq(tr.stats.npts, tr.stats.delta)
        
        # Smooth amplitude spectrum
        amplitude = np.abs(fft)
        n_smooth = int(smooth_window / (freqs[1] - freqs[0]))
        if n_smooth > 1:
            amplitude_smooth = np.convolve(amplitude, 
                                          np.ones(n_smooth)/n_smooth, 
                                          mode='same')
        else:
            amplitude_smooth = amplitude
        
        # Avoid division by zero
        amplitude_smooth[amplitude_smooth < 1e-10 * amplitude_smooth.max()] = 1.0
        
        # Whiten
        fft_whitened = fft / amplitude_smooth
        
        # Inverse FFT
        tr.data = np.fft.irfft(fft_whitened, n=tr.stats.npts)
    
    print(f"Applied spectral whitening (smooth window: {smooth_window} Hz)")
    
    return stream_whitened


def preprocessing_pipeline(stream, distances=None,
                          apply_filter=True, freqmin=5.0, freqmax=50.0,
                          apply_normalize=True, norm_method='trace',
                          apply_whiten=False, smooth_window=1.0):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    stream : obspy.Stream
        Input stream
    distances : array-like, optional
        Receiver distances
    apply_filter : bool
        Whether to apply bandpass filter
    freqmin, freqmax : float
        Filter corner frequencies
    apply_normalize : bool
        Whether to normalize
    norm_method : str
        Normalization method
    apply_whiten : bool
        Whether to apply spectral whitening
    smooth_window : float
        Whitening smooth window
    
    Returns:
    --------
    stream_processed : obspy.Stream
        Processed stream
    """
    
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    stream_processed = stream.copy()
    
    # Step 1: Bandpass filter
    if apply_filter:
        stream_processed = bandpass_filter(stream_processed, 
                                          freqmin=freqmin, 
                                          freqmax=freqmax)
    
    # Step 2: Spectral whitening (optional)
    if apply_whiten:
        stream_processed = spectral_whitening(stream_processed, 
                                             smooth_window=smooth_window)
    
    # Step 3: Normalize
    if apply_normalize:
        stream_processed = normalize_traces(stream_processed, 
                                           method=norm_method)
    
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return stream_processed


def compare_processing(stream_original, stream_processed, 
                       trace_idx=0, distances=None, save_path=None):
    """
    Compare original and processed data
    
    Parameters:
    -----------
    stream_original : obspy.Stream
        Original data
    stream_processed : obspy.Stream
        Processed data
    trace_idx : int
        Trace index to display
    distances : array-like, optional
        Receiver distances
    save_path : str, optional
        Path to save figure
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Select trace
    tr_orig = stream_original[trace_idx]
    tr_proc = stream_processed[trace_idx]
    
    time = np.arange(tr_orig.stats.npts) * tr_orig.stats.delta
    
    # Get distance info
    dist_str = ""
    if distances is not None:
        dist_str = f" (Distance: {distances[trace_idx]:.1f} m)"
    
    # Row 1: Time domain comparison
    axes[0, 0].plot(time, tr_orig.data, 'k-', linewidth=0.8)
    axes[0, 0].set_title(f'Original - Trace {trace_idx}{dist_str}', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, tr_proc.data, 'b-', linewidth=0.8)
    axes[0, 1].set_title(f'Processed - Trace {trace_idx}{dist_str}', fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Frequency domain
    fft_orig = np.fft.rfft(tr_orig.data)
    fft_proc = np.fft.rfft(tr_proc.data)
    freqs = np.fft.rfftfreq(tr_orig.stats.npts, tr_orig.stats.delta)
    
    axes[1, 0].plot(freqs, np.abs(fft_orig), 'k-', linewidth=1)
    axes[1, 0].set_title('Original - Amplitude Spectrum', fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(freqs, np.abs(fft_proc), 'b-', linewidth=1)
    axes[1, 1].set_title('Processed - Amplitude Spectrum', fontweight='bold')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 3: Shot gathers
    # Plot subset of traces for comparison
    n_display = min(12, len(stream_original))
    indices = np.linspace(0, len(stream_original)-1, n_display, dtype=int)
    
    for i, idx in enumerate(indices):
        tr_o = stream_original[idx]
        tr_p = stream_processed[idx]
        
        # Normalize for display
        norm_o = tr_o.data / np.abs(tr_o.data).max()
        norm_p = tr_p.data / np.abs(tr_p.data).max()
        
        axes[2, 0].plot(time, norm_o + i, 'k-', linewidth=0.5)
        axes[2, 1].plot(time, norm_p + i, 'b-', linewidth=0.5)
    
    axes[2, 0].set_title('Original - Shot Gather', fontweight='bold')
    axes[2, 0].set_xlabel('Time (s)', fontweight='bold')
    axes[2, 0].set_ylabel('Trace Number', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_title('Processed - Shot Gather', fontweight='bold')
    axes[2, 1].set_xlabel('Time (s)', fontweight='bold')
    axes[2, 1].set_ylabel('Trace Number', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Preprocessing Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig