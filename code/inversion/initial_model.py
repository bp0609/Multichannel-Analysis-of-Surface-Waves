# code/inversion/initial_model.py

"""
Generate initial models for inversion
"""

import numpy as np
from forward_model import LayeredEarthModel

def create_initial_model(n_layers=4, method='linear', 
                        vs_range=(200, 600), thickness_range=(2, 5)):
    """
    Create initial earth model for inversion
    
    Parameters:
    -----------
    n_layers : int
        Number of layers (including half-space)
    method : str
        'linear' - linearly increasing velocity
        'constant' - constant velocity
        'gradient' - smooth gradient
    vs_range : tuple
        (min_vs, max_vs) in m/s
    thickness_range : tuple
        (min_thickness, max_thickness) in m
    
    Returns:
    --------
    model : LayeredEarthModel
        Initial model
    """
    
    vs_min, vs_max = vs_range
    h_min, h_max = thickness_range
    
    if method == 'linear':
        # Linearly increasing velocity with depth
        vs = np.linspace(vs_min, vs_max, n_layers)
        thickness = np.ones(n_layers) * np.mean(thickness_range)
        thickness[-1] = 0  # Half-space
        
    elif method == 'constant':
        # Constant velocity
        vs = np.ones(n_layers) * np.mean(vs_range)
        thickness = np.ones(n_layers) * np.mean(thickness_range)
        thickness[-1] = 0
        
    elif method == 'gradient':
        # Smooth gradient with slight randomness
        vs = np.linspace(vs_min, vs_max, n_layers)
        vs += np.random.randn(n_layers) * (vs_max - vs_min) * 0.1
        vs = np.clip(vs, vs_min, vs_max)
        
        thickness = np.random.uniform(h_min, h_max, n_layers)
        thickness[-1] = 0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model = LayeredEarthModel(thickness, vs)
    
    return model


def create_initial_model_from_dispersion(frequencies, velocities, 
                                         n_layers=4, depth_max=50):
    """
    Create initial model estimated from dispersion curve
    Using approximate rule: Vs ≈ phase_velocity at wavelength λ
    
    Parameters:
    -----------
    frequencies : array-like
        Frequency array
    velocities : array-like
        Observed phase velocities
    n_layers : int
        Number of layers to create
    depth_max : float
        Maximum depth for model (m)
    
    Returns:
    --------
    model : LayeredEarthModel
        Initial model
    """
    
    # Calculate wavelengths
    wavelengths = velocities / frequencies
    
    # Rough depth of investigation: depth ≈ λ/3
    depths = wavelengths / 3.0
    
    # Create layer boundaries
    layer_depths = np.linspace(0, min(depth_max, depths.max()), n_layers + 1)
    thickness = np.diff(layer_depths)
    # Ensure thickness is between 2m and 5m for all layers except half-space
    thickness = np.clip(thickness, 2.0, 5.0)
    thickness = np.append(thickness, 0)  # Add half-space
    
    # Estimate Vs for each layer
    # Use velocity at the frequency corresponding to layer mid-depth
    # Note: thickness now has n_layers+1 elements (including half-space)
    vs = np.zeros(len(thickness))
    
    for i in range(len(thickness) - 1):  # All layers except half-space
        layer_mid = (layer_depths[i] + layer_depths[i+1]) / 2
        
        # Find wavelength closest to 3*layer_mid
        target_wavelength = layer_mid * 3
        idx = np.argmin(np.abs(wavelengths - target_wavelength))
        
        # Use velocity at this frequency as estimate
        # Typically Vs ≈ 0.9 * phase_velocity for fundamental mode
        # Ensure minimum velocity of 100 m/s
        vs[i] = max(100.0, velocities[idx] * 0.9)
    
    # Half-space: use velocity at lowest frequency
    vs[-1] = max(100.0, velocities[0] * 0.9)
    
    model = LayeredEarthModel(thickness, vs)
    
    return model


def define_parameter_bounds(n_layers, vs_min=100, vs_max=2000, 
                           h_min=2, h_max=30):
    """
    Define parameter bounds for inversion
    
    Parameters:
    -----------
    n_layers : int
        Number of layers
    vs_min, vs_max : float
        Bounds on Vs (m/s)
    h_min, h_max : float
        Bounds on layer thickness (m)
    
    Returns:
    --------
    bounds : dict
        Dictionary with 'vs' and 'thickness' bounds
    """
    
    bounds = {
        'vs': [(vs_min, vs_max) for _ in range(n_layers)],
        'thickness': [(h_min, h_max) for _ in range(n_layers - 1)] + [(0, 0)]  # Fix half-space
    }
    
    return bounds


if __name__ == "__main__":
    print("Testing initial model generation...")
    
    # Test 1: Linear increasing model
    model1 = create_initial_model(n_layers=5, method='linear')
    print("\nLinear Model:")
    print(model1)
    
    # Test 2: From dispersion curve
    frequencies = np.linspace(5, 50, 50)
    velocities = np.linspace(250, 500, 50)  # Synthetic
    
    model2 = create_initial_model_from_dispersion(frequencies, velocities, n_layers=4)
    print("\nModel from Dispersion:")
    print(model2)