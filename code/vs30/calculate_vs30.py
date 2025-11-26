# code/vs30/calculate_vs30.py

"""
Vs30 calculation and related site characterization parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inversion.forward_model import LayeredEarthModel

def calculate_vs30(model):
    """
    Calculate Vs30 from layered earth model
    
    Vs30 = 30 / Σ(hi/Vsi) for i=1 to n
    where hi is thickness and Vsi is shear velocity of layer i
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    
    Returns:
    --------
    vs30 : float
        Time-averaged shear wave velocity to 30m depth (m/s)
    """
    
    depth = 0
    travel_time = 0
    
    for i in range(model.n_layers):
        if depth >= 30.0:
            break
        
        if model.thickness[i] == 0:  # Half-space
            remaining_depth = 30.0 - depth
            travel_time += remaining_depth / model.vs[i]
            break
        else:
            layer_bottom = depth + model.thickness[i]
            
            if layer_bottom <= 30.0:
                # Full layer within 30m
                travel_time += model.thickness[i] / model.vs[i]
                depth = layer_bottom
            else:
                # Partial layer
                remaining_thickness = 30.0 - depth
                travel_time += remaining_thickness / model.vs[i]
                depth = 30.0
                break
    
    vs30 = 30.0 / travel_time
    return vs30


def compute_vs_time_average(model_layers, depth=30.0):
    """
    Compute time-averaged Vs to arbitrary depth from layer list
    Alternative interface that accepts raw layer data
    
    Parameters:
    -----------
    model_layers : list of tuples
        List of (thickness, Vs) tuples. Last layer can have thickness=None or np.inf for half-space.
    depth : float
        Target depth in meters (default: 30.0)
    
    Returns:
    --------
    vsz : float
        Time-averaged shear wave velocity to depth (m/s)
    """
    remaining = depth
    travel_time = 0.0
    
    for h, vs in model_layers:
        if remaining <= 1e-8:
            break
            
        if h is None or (isinstance(h, float) and np.isinf(h)) or h == 0:
            # Half-space - use all remaining depth
            h_eff = remaining
        else:
            # Regular layer - use minimum of layer thickness and remaining depth
            h_eff = min(h, remaining)
        
        travel_time += h_eff / vs
        remaining -= h_eff
    
    # If depth not fully covered, extrapolate using last layer's Vs
    if remaining > 1e-6:
        last_vs = model_layers[-1][1]
        travel_time += remaining / last_vs
    
    vsz = depth / travel_time
    return vsz


def calculate_vsz(model, depth_target):
    """
    Calculate time-averaged Vs to arbitrary depth
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    depth_target : float
        Target depth (m)
    
    Returns:
    --------
    vsz : float
        Time-averaged shear wave velocity to depth_target (m/s)
    """
    
    depth = 0
    travel_time = 0
    
    for i in range(model.n_layers):
        if depth >= depth_target:
            break
        
        if model.thickness[i] == 0:  # Half-space
            remaining_depth = depth_target - depth
            travel_time += remaining_depth / model.vs[i]
            break
        else:
            layer_bottom = depth + model.thickness[i]
            
            if layer_bottom <= depth_target:
                travel_time += model.thickness[i] / model.vs[i]
                depth = layer_bottom
            else:
                remaining_thickness = depth_target - depth
                travel_time += remaining_thickness / model.vs[i]
                depth = depth_target
                break
    
    vsz = depth_target / travel_time
    return vsz


def calculate_depth_to_bedrock(model, bedrock_threshold=760):
    """
    Estimate depth to bedrock (Vs > threshold)
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    bedrock_threshold : float
        Vs threshold for bedrock definition (m/s)
    
    Returns:
    --------
    depth_to_bedrock : float
        Depth to bedrock interface (m), or None if not found
    """
    
    cumulative_depth = 0
    
    for i in range(model.n_layers):
        if model.vs[i] >= bedrock_threshold:
            return cumulative_depth
        
        if model.thickness[i] == 0:  # Half-space
            return None  # Bedrock not reached
        
        cumulative_depth += model.thickness[i]
    
    return None


def calculate_vs_statistics(model, max_depth=50):
    """
    Calculate various Vs statistics for the profile
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    max_depth : float
        Maximum depth for analysis (m)
    
    Returns:
    --------
    stats : dict
        Dictionary with various statistics
    """
    
    stats = {
        'vs30': calculate_vs30(model),
        'vs10': calculate_vsz(model, 10.0),
        'vs15': calculate_vsz(model, 15.0),
        'vs20': calculate_vsz(model, 20.0),
        'vs_min': np.min(model.vs),
        'vs_max': np.max(model.vs),
        'vs_surface': model.vs[0],
        'depth_to_bedrock': calculate_depth_to_bedrock(model, 760)
    }
    
    return stats


def propagate_vs30_uncertainty(acceptable_models):
    """
    Calculate Vs30 uncertainty from ensemble of acceptable models
    
    Parameters:
    -----------
    acceptable_models : list
        List of acceptable LayeredEarthModel objects
    
    Returns:
    --------
    vs30_mean : float
        Mean Vs30
    vs30_std : float
        Standard deviation of Vs30
    vs30_values : array
        All Vs30 values
    """
    
    vs30_values = np.array([calculate_vs30(m) for m in acceptable_models])
    
    vs30_mean = np.mean(vs30_values)
    vs30_std = np.std(vs30_values)
    
    return vs30_mean, vs30_std, vs30_values


def format_vs30(vs30_value, decimal_places=1):
    """
    Format Vs30 value with consistent rounding for reporting
    
    This ensures all printed and saved Vs30 values use the same formatting,
    preventing inconsistencies like 287.5 vs 287.6 vs 288.
    
    Parameters:
    -----------
    vs30_value : float
        Raw Vs30 value
    decimal_places : int
        Number of decimal places (default: 1)
    
    Returns:
    --------
    vs30_formatted : float
        Consistently rounded Vs30 value
    """
    return round(vs30_value, decimal_places)


if __name__ == "__main__":
    # Test
    from code.inversion.forward_model import LayeredEarthModel
    
    thickness = [10.0, 15.0, 0.0]
    vs = [250.0, 400.0, 600.0]
    
    model = LayeredEarthModel(thickness, vs)
    
    print("Test Model:")
    print(model)
    
    stats = calculate_vs_statistics(model)
    print("\nVs Statistics:")
    for key, value in stats.items():
        if value is not None:
            if 'vs' in key.lower():
                # Use consistent formatting for all Vs values
                formatted_value = format_vs30(value)
                print(f"  {key}: {formatted_value:.1f} m/s")
            else:
                print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: Not found")
    
    # Demonstrate consistent formatting
    print("\nConsistent Vs30 formatting examples:")
    vs30_raw = calculate_vs30(model)
    print(f"  Raw Vs30: {vs30_raw}")
    print(f"  Formatted Vs30: {format_vs30(vs30_raw):.1f} m/s")