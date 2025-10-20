# code/inversion/least_squares.py

"""
Damped Least-Squares Inversion for MASW
"""

import numpy as np
from scipy.optimize import least_squares
from forward_model import LayeredEarthModel, compute_dispersion_curve

def objective_function(params, n_layers, frequencies, observed_velocities, 
                      weights=None, fix_vp_vs_ratio=True):
    """
    Objective function for least-squares inversion
    
    Parameters:
    -----------
    params : array-like
        Model parameters [vs1, vs2, ..., h1, h2, ...]
    n_layers : int
        Number of layers
    frequencies : array-like
        Frequency array
    observed_velocities : array-like
        Observed phase velocities
    weights : array-like, optional
        Data weights (inverse of uncertainties)
    fix_vp_vs_ratio : bool
        If True, Vp calculated from Vs
    
    Returns:
    --------
    residuals : ndarray
        Weighted residuals
    """
    
    # Unpack parameters
    vs = params[:n_layers]
    thickness = np.append(params[n_layers:], 0)  # Add zero for half-space
    
    # Create model
    model = LayeredEarthModel(thickness, vs)
    
    # Compute theoretical dispersion curve
    try:
        theoretical_velocities = compute_dispersion_curve(model, frequencies)
    except:
        # Return large residual if forward modeling fails
        return np.ones_like(observed_velocities) * 1e6
    
    # Calculate residuals
    residuals = observed_velocities - theoretical_velocities
    
    # Remove NaN values
    valid = ~np.isnan(residuals)
    residuals = residuals[valid]
    
    # Apply weights if provided
    if weights is not None:
        weights_valid = weights[valid]
        residuals *= weights_valid
    
    return residuals


def invert_least_squares(frequencies, observed_velocities, initial_model,
                        bounds=None, uncertainties=None, max_iterations=100,
                        damping=0.01, verbose=True):
    """
    Perform damped least-squares inversion
    
    Parameters:
    -----------
    frequencies : array-like
        Frequency array
    observed_velocities : array-like
        Observed phase velocities
    initial_model : LayeredEarthModel
        Initial model guess
    bounds : dict, optional
        Parameter bounds {'vs': [...], 'thickness': [...]}
    uncertainties : array-like, optional
        Data uncertainties (standard deviations)
    max_iterations : int
        Maximum number of iterations
    damping : float
        Damping factor (Levenberg-Marquardt)
    verbose : bool
        Print progress
    
    Returns:
    --------
    final_model : LayeredEarthModel
        Inverted model
    result : OptimizeResult
        Optimization result object
    """
    
    if verbose:
        print("=" * 60)
        print("DAMPED LEAST-SQUARES INVERSION")
        print("=" * 60)
    
    n_layers = initial_model.n_layers
    
    # Prepare initial parameters
    x0 = np.concatenate([initial_model.vs, initial_model.thickness[:-1]])
    
    # Prepare weights from uncertainties
    weights = None
    if uncertainties is not None:
        weights = 1.0 / uncertainties
        weights /= np.mean(weights)  # Normalize
    
    # Prepare bounds
    if bounds is None:
        lower = np.concatenate([
            np.ones(n_layers) * 100,  # Vs min
            np.ones(n_layers - 1) * 2   # thickness min
        ])
        upper = np.concatenate([
            np.ones(n_layers) * 2000,  # Vs max
            np.ones(n_layers - 1) * 50   # thickness max
        ])
    else:
        vs_bounds = np.array(bounds['vs'])
        h_bounds = np.array(bounds['thickness'][:-1])  # Exclude half-space
        
        lower = np.concatenate([vs_bounds[:, 0], h_bounds[:, 0]])
        upper = np.concatenate([vs_bounds[:, 1], h_bounds[:, 1]])
    
    # Run optimization
    if verbose:
        print(f"\nInitial Model:")
        print(initial_model)
        print(f"\nStarting optimization...")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Damping: {damping}")
    
    result = least_squares(
        objective_function,
        x0,
        args=(n_layers, frequencies, observed_velocities, weights),
        bounds=(lower, upper),
        max_nfev=max_iterations * len(x0),
        verbose=2 if verbose else 0,
        method='trf',  # Trust Region Reflective
        ftol=1e-6,
        xtol=1e-6
    )
    
    # Extract final model
    final_vs = result.x[:n_layers]
    final_thickness = np.append(result.x[n_layers:], 0)
    final_model = LayeredEarthModel(final_thickness, final_vs)
    
    # Calculate RMS error
    final_predicted = compute_dispersion_curve(final_model, frequencies)
    residuals = observed_velocities - final_predicted
    rms_error = np.sqrt(np.mean(residuals**2))
    
    if verbose:
        print(f"\n" + "=" * 60)
        print("INVERSION COMPLETE")
        print("=" * 60)
        print(f"\nFinal Model:")
        print(final_model)
        print(f"\nRMS Error: {rms_error:.2f} m/s")
        print(f"Iterations: {result.nfev}")
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Message: {result.message}")
    
    return final_model, result, rms_error