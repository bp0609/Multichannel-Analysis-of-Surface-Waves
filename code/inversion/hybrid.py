# code/inversion/hybrid.py

"""
Hybrid inversion: Global search + Local refinement
"""

import numpy as np
from global_search import monte_carlo_inversion
from least_square import invert_least_squares

def hybrid_inversion(frequencies, observed_velocities, n_layers=4,
                    n_monte_carlo=500, bounds=None, uncertainties=None,
                    verbose=True):
    """
    Two-stage hybrid inversion:
    1. Monte Carlo global search
    2. Least-squares local refinement
    
    Parameters:
    -----------
    frequencies : array-like
        Frequency array
    observed_velocities : array-like
        Observed phase velocities
    n_layers : int
        Number of layers
    n_monte_carlo : int
        Number of Monte Carlo samples
    bounds : dict, optional
        Parameter bounds
    uncertainties : array-like, optional
        Data uncertainties
    verbose : bool
        Print progress
    
    Returns:
    --------
    final_model : LayeredEarthModel
        Final inverted model
    mc_model : LayeredEarthModel
        Best model from Monte Carlo
    rms_error : float
        Final RMS error
    """
    
    if verbose:
        print("=" * 60)
        print("HYBRID INVERSION (Monte Carlo + Least Squares)")
        print("=" * 60)
    
    # Stage 1: Monte Carlo global search
    if verbose:
        print("\nSTAGE 1: Global Search")
        print("-" * 60)
    
    mc_model, mc_misfit, all_models, all_misfits = monte_carlo_inversion(
        frequencies, observed_velocities,
        n_models=n_monte_carlo,
        n_layers=n_layers,
        bounds=bounds,
        uncertainties=uncertainties,
        verbose=verbose
    )
    
    # Stage 2: Least-squares refinement
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: Local Refinement")
        print("-" * 60)
    
    final_model, result, rms_error = invert_least_squares(
        frequencies, observed_velocities,
        initial_model=mc_model,
        bounds=bounds,
        uncertainties=uncertainties,
        max_iterations=100,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("HYBRID INVERSION COMPLETE")
        print("=" * 60)
        print(f"\nMonte Carlo RMS: {mc_misfit:.2f} m/s")
        print(f"Final RMS: {rms_error:.2f} m/s")
        print(f"Improvement: {((mc_misfit - rms_error) / mc_misfit * 100):.1f}%")
    
    return final_model, mc_model, rms_error