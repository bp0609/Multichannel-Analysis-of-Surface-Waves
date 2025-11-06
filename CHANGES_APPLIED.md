# Changes Applied to Fix MASW Inversion Issues

This document summarizes the changes made to address the uniform layer thickness and inconsistent Vs30 reporting issues.

## Date Applied
November 6, 2025

## Changes Made

### 1. Hybrid Inversion (`code/inversion/hybrid.py`)

**Problem:** The hybrid inversion was unconditionally using the refined model even when it had a worse RMS than the Monte Carlo seed model.

**Solution:** Added logic to compare RMS before and after local refinement:
- Store the Monte Carlo seed model and its RMS
- Run least-squares refinement
- Compare seed_rms vs refined_rms
- Only accept refinement if it improves (with small tolerance of 1e-6)
- Keep the better model and report which was chosen

**Key Changes:**
```python
# Accept refinement only if it improves RMS
if refined_rms + improvement_tol < seed_rms:
    final_model = refined_model
    final_rms = refined_rms
    accepted = "refined"
else:
    final_model = seed_model
    final_rms = seed_rms
    accepted = "seed"
```

### 2. Final Model Selection (`code/inversion/run_inversion.py`)

**Problem:** The code was always choosing the hybrid model as final, even when other methods had better RMS.

**Solution:** Implemented explicit minimum-RMS selection:
- Create a dictionary of all methods and their RMS values
- Find the method with the minimum RMS
- Select that as the final model
- Report which method was chosen and all RMS values
- Use consistent Vs30 calculation from `calculate_vs30` module

**Key Changes:**
```python
results = {
    "least_squares": {"model": model_ls, "rms": rms_ls},
    "monte_carlo_best": {"model": model_mc, "rms": misfit_mc},
    "hybrid": {"model": model_hybrid, "rms": rms_hybrid}
}

rms_items = [(k, v["rms"]) for k, v in results.items()]
best_method, best_rms = min(rms_items, key=lambda x: x[1])
final_model = results[best_method]["model"]
```

### 3. Monte Carlo Constraints (`code/inversion/global_search.py`)

**Problem:** Monte Carlo was generating unrealistic models with very thick shallow layers (>30m in first layer).

**Solution:** Added parameter constraints and penalties:
- Limit first layer thickness to maximum of 40m
- Add penalty for models where >90% of top 30m is a single layer
- Penalty is 10% of RMS when triggered

**Key Changes:**
```python
# Constrain first layer
max_first_layer_thickness = min(40.0, h_max)
thickness[0] = np.random.uniform(h_min, max_first_layer_thickness)

# Add penalty for unrealistic distributions
frac_from_first_layer = depth_30m / 30.0 if thickness[0] >= 30.0 else thickness[0] / 30.0
penalty = 0.0
if frac_from_first_layer > 0.9:
    penalty = 0.1 * rms
misfit = rms + penalty
```

### 4. Consistent Vs30 Formatting (`code/vs30/calculate_vs30.py`)

**Problem:** Vs30 values were being formatted inconsistently (287.5, 287.6, 288) in different parts of the code.

**Solution:** Added helper functions for consistent formatting:
- Added `compute_vs_time_average()` function as alternative interface
- Added `format_vs30()` function to ensure consistent rounding
- Updated test code to demonstrate usage
- All Vs30 values now use single decimal place rounding

**Key Changes:**
```python
def format_vs30(vs30_value, decimal_places=1):
    """
    Format Vs30 value with consistent rounding for reporting
    """
    return round(vs30_value, decimal_places)
```

## Expected Outcomes

After these changes:

1. **Better Model Selection:** The final model will always be the one with the lowest RMS error
2. **No Worse Refinement:** Hybrid inversion won't accept local refinement that increases error
3. **More Realistic Layering:** Monte Carlo will avoid generating models with unrealistically thick shallow layers
4. **Consistent Reporting:** All Vs30 values will be formatted consistently throughout the code and output files

## Testing Checklist

To verify the fixes work:

- [x] Applied hybrid.py changes - guards against worse refinement
- [x] Applied run_inversion.py changes - selects minimum RMS model
- [x] Applied global_search.py changes - constrains layer parameters
- [x] Applied calculate_vs30.py changes - consistent rounding

## Next Steps

Run the inversion again with:
```bash
python code/inversion/run_inversion.py
```

Check:
1. Which method is chosen as final and its RMS
2. Whether all RMS values are reported correctly
3. Whether the final Vs30 is consistent across all output files
4. Whether the layer thicknesses are more realistic (multi-layer top 30m)

## Notes

- The lint errors about imports in `run_inversion.py` and `calculate_vs30.py` are expected - the imports are added dynamically with `sys.path.append`
- The penalty factor (0.1) and threshold (0.9) in global_search.py can be adjusted if needed
- The max_first_layer_thickness (40m) can be adjusted based on site-specific requirements
