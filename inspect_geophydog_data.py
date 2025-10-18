# inspect_geophydog_data.py
import os
import glob
from obspy import read

data_dir = "data/raw/geophydog"

# List all SAC files
sac_files = sorted(glob.glob(os.path.join(data_dir, "*.SAC")))
print(f"Found {len(sac_files)} SAC files")

# Read first seismogram
if sac_files:
    st = read(sac_files[0])
    tr = st[0]
    
    print("\n--- Seismogram Information ---")
    print(f"Station: {tr.stats.station}")
    print(f"Sampling rate: {tr.stats.sampling_rate} Hz")
    print(f"Number of samples: {tr.stats.npts}")
    print(f"Duration: {tr.stats.npts / tr.stats.sampling_rate:.2f} seconds")
    print(f"Delta (sample interval): {tr.stats.delta} seconds")
    
    # Check for distance information
    if hasattr(tr.stats.sac, 'dist'):
        print(f"Distance from source: {tr.stats.sac.dist} m")
    
# Look for dispersion curve files
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
if txt_files:
    print(f"\n--- Found {len(txt_files)} text files (likely dispersion curves) ---")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")