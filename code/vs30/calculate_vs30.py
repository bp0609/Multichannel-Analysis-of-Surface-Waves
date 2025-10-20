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
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: Not found")