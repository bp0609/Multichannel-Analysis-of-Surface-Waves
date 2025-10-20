# code/inversion/forward_model.py

"""
Forward modeling for MASW dispersion curves
Calculate theoretical dispersion curves from layered earth models
"""

import numpy as np
from disba import PhaseDispersion, GroupDispersion
import matplotlib.pyplot as plt

class LayeredEarthModel:
    """
    Class to represent a layered earth model
    """
    
    def __init__(self, thickness, vs, vp=None, rho=None):
        """
        Initialize layered earth model
        
        Parameters:
        -----------
        thickness : array-like
            Layer thicknesses in meters (last layer = 0 for half-space)
        vs : array-like
            Shear wave velocities in m/s
        vp : array-like, optional
            P-wave velocities in m/s (if None, use Vp/Vs = 1.73)
        rho : array-like, optional
            Densities in g/cm³ (if None, use Gardner's relation)
        """
        
        self.thickness = np.array(thickness)
        self.vs = np.array(vs)
        
        # Ensure last layer has zero thickness (half-space)
        if self.thickness[-1] != 0:
            print("Warning: Setting last layer thickness to 0 (half-space)")
            self.thickness[-1] = 0
        
        # Calculate Vp if not provided (Poisson's ratio ~ 0.25)
        if vp is None:
            self.vp = self.vs * 1.73  # Typical Vp/Vs ratio
        else:
            self.vp = np.array(vp)
        
        # Calculate density if not provided (Gardner's relation)
        if rho is None:
            # Gardner: rho = 0.31 * Vp^0.25 (Vp in m/s, rho in g/cm³)
            self.rho = 0.31 * (self.vp / 1000) ** 0.25
        else:
            self.rho = np.array(rho)
        
        self.n_layers = len(self.thickness)
    
    def get_depth_array(self, dz=0.5):
        """
        Get depth array and Vs profile for plotting
        
        Parameters:
        -----------
        dz : float
            Depth interval (m)
        
        Returns:
        --------
        depths : ndarray
            Depth array
        vs_profile : ndarray
            Vs at each depth
        """
        
        # Calculate total depth (exclude half-space)
        total_depth = np.sum(self.thickness[:-1])
        
        if total_depth == 0:
            total_depth = 50  # Default for single half-space
        
        depths = np.arange(0, total_depth + dz, dz)
        vs_profile = np.zeros_like(depths)
        
        cumulative_depth = 0
        for i in range(self.n_layers - 1):  # Exclude half-space
            layer_top = cumulative_depth
            layer_bottom = cumulative_depth + self.thickness[i]
            
            mask = (depths >= layer_top) & (depths < layer_bottom)
            vs_profile[mask] = self.vs[i]
            
            cumulative_depth = layer_bottom
        
        # Fill remaining with half-space velocity
        vs_profile[depths >= cumulative_depth] = self.vs[-1]
        
        return depths, vs_profile
    
    def calculate_vs30(self):
        """
        Calculate Vs30 from the model
        
        Returns:
        --------
        vs30 : float
            Time-averaged shear wave velocity to 30m depth
        """
        
        depth = 0
        travel_time = 0
        
        for i in range(self.n_layers):
            if depth >= 30.0:
                break
            
            if self.thickness[i] == 0:  # Half-space
                remaining_depth = 30.0 - depth
                travel_time += remaining_depth / self.vs[i]
                break
            else:
                layer_bottom = depth + self.thickness[i]
                
                if layer_bottom <= 30.0:
                    # Full layer within 30m
                    travel_time += self.thickness[i] / self.vs[i]
                    depth = layer_bottom
                else:
                    # Partial layer
                    remaining_thickness = 30.0 - depth
                    travel_time += remaining_thickness / self.vs[i]
                    depth = 30.0
                    break
        
        vs30 = 30.0 / travel_time
        return vs30
    
    def __str__(self):
        """String representation of model"""
        s = "Layered Earth Model:\n"
        s += "-" * 60 + "\n"
        s += f"{'Layer':<8} {'Thickness(m)':<15} {'Vs(m/s)':<12} {'Vp(m/s)':<12} {'Rho(g/cm³)':<12}\n"
        s += "-" * 60 + "\n"
        
        for i in range(self.n_layers):
            thick_str = f"{self.thickness[i]:.2f}" if self.thickness[i] > 0 else "∞ (half-space)"
            s += f"{i+1:<8} {thick_str:<15} {self.vs[i]:<12.1f} {self.vp[i]:<12.1f} {self.rho[i]:<12.2f}\n"
        
        s += "-" * 60 + "\n"
        s += f"Vs30 = {self.calculate_vs30():.1f} m/s\n"
        
        return s


def compute_dispersion_curve(model, frequencies, wave='rayleigh', mode=0, 
                             velocity_type='phase'):
    """
    Compute theoretical dispersion curve for a given model
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    frequencies : array-like
        Frequencies in Hz
    wave : str
        'rayleigh' or 'love'
    mode : int
        Mode number (0=fundamental, 1=first higher, etc.)
    velocity_type : str
        'phase' or 'group'
    
    Returns:
    --------
    velocities : ndarray
        Phase or group velocities in m/s (NaN where mode doesn't exist)
    """
    
    # Convert to km and km/s for disba
    thickness_km = model.thickness / 1000.0
    vs_kms = model.vs / 1000.0
    vp_kms = model.vp / 1000.0
    
    # Convert frequencies to periods (must be sorted ascending)
    periods = 1.0 / frequencies
    # Sort periods in ascending order
    sort_idx = np.argsort(periods)
    periods_sorted = periods[sort_idx]
    
    try:
        if velocity_type == 'phase':
            pd = PhaseDispersion(*[thickness_km, vp_kms, vs_kms, model.rho])
            result = pd(periods_sorted, mode=mode, wave=wave)
        else:  # group
            gd = GroupDispersion(*[thickness_km, vp_kms, vs_kms, model.rho])
            result = gd(periods_sorted, mode=mode, wave=wave)
        
        # Convert back to m/s and restore original frequency order
        velocities_sorted = result.velocity * 1000.0
        velocities = np.zeros_like(frequencies)
        velocities[sort_idx] = velocities_sorted
        
    except Exception as e:
        print(f"Error computing dispersion curve: {e}")
        velocities = np.full_like(frequencies, np.nan)
    
    return velocities


def plot_model_and_dispersion(model, frequencies, observed_velocities=None,
                              save_path=None):
    """
    Plot earth model and corresponding dispersion curve
    
    Parameters:
    -----------
    model : LayeredEarthModel
        Earth model
    frequencies : array-like
        Frequency array
    observed_velocities : array-like, optional
        Observed dispersion curve for comparison
    save_path : str, optional
        Path to save figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Vs profile
    ax = axes[0]
    depths, vs_profile = model.get_depth_array(dz=0.5)
    
    ax.plot(vs_profile, depths, 'b-', linewidth=2.5, label='Vs Profile')
    
    # Mark layer boundaries
    cumulative_depth = 0
    for i in range(model.n_layers - 1):
        cumulative_depth += model.thickness[i]
        ax.axhline(cumulative_depth, color='gray', linestyle='--', 
                   alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Vs (m/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('Shear Wave Velocity Profile', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(model.vs) * 1.1)
    
    # Add Vs30 annotation
    vs30 = model.calculate_vs30()
    ax.axhline(30, color='red', linestyle=':', linewidth=2, label='30m depth')
    ax.text(0.98, 0.02, f'Vs30 = {vs30:.1f} m/s', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.legend(loc='lower right')
    
    # Plot 2: Dispersion curve
    ax = axes[1]
    
    # Theoretical curve
    theoretical_velocities = compute_dispersion_curve(model, frequencies)
    ax.plot(frequencies, theoretical_velocities, 'b-', 
            linewidth=2.5, label='Theoretical (Model)', marker='o', markersize=4)
    
    # Observed curve if provided
    if observed_velocities is not None:
        ax.plot(frequencies, observed_velocities, 'ro', 
                markersize=6, label='Observed', alpha=0.7)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phase Velocity (m/s)', fontsize=12, fontweight='bold')
    ax.set_title('Dispersion Curve', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    return fig


def test_forward_model():
    """
    Test forward modeling with a simple 3-layer model
    """
    
    print("=" * 60)
    print("TESTING FORWARD MODEL")
    print("=" * 60)
    
    # Create simple 3-layer model
    thickness = [10.0, 20.0, 0.0]  # meters, last is half-space
    vs = [200.0, 400.0, 600.0]     # m/s
    
    model = LayeredEarthModel(thickness, vs)
    print(model)
    
    # Compute dispersion curve
    frequencies = np.linspace(5, 50, 50)
    velocities = compute_dispersion_curve(model, frequencies)
    
    print(f"\nComputed dispersion curve:")
    print(f"  Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")
    print(f"  Velocity range: {np.nanmin(velocities):.1f} - {np.nanmax(velocities):.1f} m/s")
    
    # Plot
    plot_model_and_dispersion(model, frequencies)
    
    return model, frequencies, velocities


if __name__ == "__main__":
    test_forward_model()