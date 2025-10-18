# code/data_loading/load_sac.py
"""
Functions for loading SAC format seismic data
"""
import os
import glob
from obspy import read, Stream
import numpy as np

def load_geophydog_data(data_dir):
    """
    Load all SAC files from Geophydog directory
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing SAC files
    
    Returns:
    --------
    stream : obspy.Stream
        Stream object containing all traces
    """
    sac_files = sorted(glob.glob(os.path.join(data_dir, "*.sac")))
    
    if not sac_files:
        raise FileNotFoundError(f"No SAC files found in {data_dir}")
    
    print(f"Loading {len(sac_files)} SAC files...")
    stream = Stream()
    
    for sac_file in sac_files:
        st = read(sac_file)
        stream += st
    
    print(f"Loaded {len(stream)} traces")
    return stream

def get_acquisition_geometry(stream):
    """
    Extract acquisition geometry from stream
    
    Parameters:
    -----------
    stream : obspy.Stream
        Stream containing traces
    
    Returns:
    --------
    geometry : dict
        Dictionary with source_offset, receiver_spacing, distances
    """
    distances = []
    
    for tr in stream:
        if hasattr(tr.stats.sac, 'dist'):
            distances.append(tr.stats.sac.dist)
        elif hasattr(tr.stats, 'distance'):
            distances.append(tr.stats.distance)
    
    if not distances:
        print("Warning: No distance information found in headers")
        return None
    
    distances = np.array(distances)
    
    geometry = {
        'distances': distances,
        'receiver_spacing': np.median(np.diff(sorted(distances))),
        'source_offset': min(distances),
        'n_receivers': len(distances)
    }
    
    return geometry

if __name__ == "__main__":
    # Test the loader
    import sys
    sys.path.append('..')
    from config import RAW_DATA_DIR
    
    stream = load_geophydog_data(RAW_DATA_DIR)
    geometry = get_acquisition_geometry(stream)
    
    if geometry:
        print("\n--- Acquisition Geometry ---")
        for key, value in geometry.items():
            print(f"{key}: {value}")