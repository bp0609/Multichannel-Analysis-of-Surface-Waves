# test_disba.py
import numpy as np
from disba import PhaseDispersion

# Define a simple 3-layer velocity model
# thickness (km), Vp (km/s), Vs (km/s), density (g/cm³)
velocity_model = np.array([
    [0.01, 0.50, 0.25, 1.80],  # 10m soft soil layer
    [0.02, 1.50, 0.75, 2.00],  # 20m medium soil
    [0.00, 3.00, 1.50, 2.20],  # Half-space (infinite)
])

# Define frequency range (periods)
periods = np.logspace(-1, 1, 50)  # 0.1 to 10 seconds

# Calculate fundamental mode Rayleigh wave dispersion
pd = PhaseDispersion(*velocity_model.T)
result = pd(periods, mode=0, wave="rayleigh")

print("Dispersion calculation successful!")
print(f"Period range: {result.period[0]:.2f} to {result.period[-1]:.2f} seconds")
print(f"Velocity range: {result.velocity.min():.2f} to {result.velocity.max():.2f} km/s")