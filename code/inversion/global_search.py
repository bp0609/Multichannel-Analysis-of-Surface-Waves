# code/inversion/global_search.py

"""
Global search inversion methods (Monte Carlo, Genetic Algorithm)
"""

import numpy as np
from forward_model import LayeredEarthModel, compute_dispersion_curve
from tqdm import tqdm

def monte_carlo_inversion(frequencies, observed_velocities, n_models=1000,
                         n_layers=4, bounds=None, uncertainties=None,
                         verbose=True):
    """
    Monte Carlo search for best-fitting model
    
    Parameters:
    -----------
    frequencies : array-like
        Frequency array
    observed_velocities : array-like
        Observed phase velocities
    n_models : int
        Number of random models to test
    n_layers : int
        Number of layers
    bounds : dict, optional
        Parameter bounds
    uncertainties : array-like, optional
        Data uncertainties
    verbose : bool
        Print progress
    
    Returns:
    --------
    best_model : LayeredEarthModel
        Best-fitting model
    best_misfit : float
        RMS misfit of best model
    all_models : list
        List of all tested models
    all_misfits : array
        Misfits for all models
    """
    
    if verbose:
        print("=" * 60)
        print("MONTE CARLO INVERSION")
        print("=" * 60)
        print(f"Testing {n_models} random models...")
    
    # Set default bounds
    if bounds is None:
        vs_min, vs_max = 100, 2000
        h_min, h_max = 2, 50
    else:
        vs_bounds = np.array(bounds['vs'])
        h_bounds = np.array(bounds['thickness'][:-1])
        vs_min, vs_max = vs_bounds[0]
        h_min, h_max = h_bounds[0]
    
    # Calculate weights
    weights = None
    if uncertainties is not None:
        weights = 1.0 / (uncertainties ** 2)
        weights /= np.sum(weights)
    else:
        weights = np.ones_like(observed_velocities) / len(observed_velocities)
    
    all_models = []
    all_misfits = []
    
    best_misfit = np.inf
    best_model = None
    
    # Monte Carlo sampling
    for i in tqdm(range(n_models), desc="Sampling models", disable=not verbose):
        # Generate random model
        vs = np.random.uniform(vs_min, vs_max, n_layers)
        vs = np.sort(vs)  # Ensure increasing velocity (optional)
        
        thickness = np.random.uniform(h_min, h_max, n_layers)
        thickness[-1] = 0  # Half-space
        
        model = LayeredEarthModel(thickness, vs)
        
        # Compute dispersion curve
        try:
            predicted = compute_dispersion_curve(model, frequencies)
            
            # Calculate weighted RMS misfit
            residuals = observed_velocities - predicted
            valid = ~np.isnan(residuals)
            
            if np.sum(valid) > 0:
                misfit = np.sqrt(np.sum(weights[valid] * residuals[valid]**2))
            else:
                misfit = np.inf
        except:
            misfit = np.inf
        
        all_models.append(model)
        all_misfits.append(misfit)
        
        # Update best model
        if misfit < best_misfit:
            best_misfit = misfit
            best_model = model
    
    all_misfits = np.array(all_misfits)
    
    if verbose:
        print(f"\nBest model found:")
        print(best_model)
        print(f"RMS Misfit: {best_misfit:.2f} m/s")
        print(f"Mean misfit: {np.mean(all_misfits):.2f} m/s")
        print(f"Std misfit: {np.std(all_misfits):.2f} m/s")
    
    return best_model, best_misfit, all_models, all_misfits


def analyze_monte_carlo_results(all_models, all_misfits, threshold_percentile=10):
    """
    Analyze Monte Carlo results to assess uncertainty
    
    Parameters:
    -----------
    all_models : list
        List of all tested models
    all_misfits : array
        Misfits for all models
    threshold_percentile : float
        Percentile for acceptable models
    
    Returns:
    --------
    acceptable_models : list
        Models within threshold
    vs_ranges : ndarray
        (min, max) Vs for each layer among acceptable models
    """
    
    # Find acceptable models
    threshold = np.percentile(all_misfits, threshold_percentile)
    acceptable_idx = all_misfits <= threshold
    
    acceptable_models = [all_models[i] for i in range(len(all_models)) 
                        if acceptable_idx[i]]
    
    print(f"\nAcceptable models (within {threshold_percentile}th percentile):")
    print(f"  Threshold misfit: {threshold:.2f} m/s")
    print(f"  Number of models: {len(acceptable_models)}")
    
    # Analyze Vs ranges
    n_layers = all_models[0].n_layers
    vs_ranges = np.zeros((n_layers, 2))
    
    for i in range(n_layers):
        vs_values = [m.vs[i] for m in acceptable_models]
        vs_ranges[i, 0] = np.min(vs_values)
        vs_ranges[i, 1] = np.max(vs_values)
        
        print(f"  Layer {i+1} Vs range: {vs_ranges[i, 0]:.1f} - {vs_ranges[i, 1]:.1f} m/s")
    
    return acceptable_models, vs_ranges