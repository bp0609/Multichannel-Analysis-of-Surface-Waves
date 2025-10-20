# code/vs30/site_classification.py

"""
NEHRP and other site classification systems
"""

import numpy as np

class NEHRPClassification:
    """
    NEHRP site classification based on Vs30
    """
    
    # NEHRP 2020 (ASCE 7-22) site class boundaries
    SITE_CLASSES = {
        'A': {'vs30_min': 1500, 'vs30_max': np.inf, 'description': 'Hard Rock'},
        'B': {'vs30_min': 760, 'vs30_max': 1500, 'description': 'Rock'},
        'C': {'vs30_min': 360, 'vs30_max': 760, 'description': 'Very Dense Soil and Soft Rock'},
        'D': {'vs30_min': 180, 'vs30_max': 360, 'description': 'Stiff Soil'},
        'E': {'vs30_min': 0, 'vs30_max': 180, 'description': 'Soft Soil'},
    }
    
    # Extended ASCE 7-22 classes
    EXTENDED_CLASSES = {
        'A': {'vs30_min': 1500, 'vs30_max': np.inf, 'description': 'Hard Rock'},
        'AB': {'vs30_min': 1200, 'vs30_max': 1500, 'description': 'Hard Rock to Rock'},
        'B': {'vs30_min': 1000, 'vs30_max': 1200, 'description': 'Rock'},
        'BC': {'vs30_min': 700, 'vs30_max': 1000, 'description': 'Rock to Very Dense Soil'},
        'C': {'vs30_min': 450, 'vs30_max': 700, 'description': 'Very Dense Soil and Soft Rock'},
        'CD': {'vs30_min': 350, 'vs30_max': 450, 'description': 'Dense Soil'},
        'D': {'vs30_min': 200, 'vs30_max': 350, 'description': 'Stiff Soil'},
        'DE': {'vs30_min': 150, 'vs30_max': 200, 'description': 'Stiff to Medium Soil'},
        'E': {'vs30_min': 0, 'vs30_max': 150, 'description': 'Soft Soil'},
    }
    
    @staticmethod
    def classify(vs30, extended=False):
        """
        Classify site based on Vs30
        
        Parameters:
        -----------
        vs30 : float
            Vs30 value (m/s)
        extended : bool
            Use extended classification (ASCE 7-22 9 classes)
        
        Returns:
        --------
        site_class : str
            NEHRP site class letter
        description : str
            Site class description
        """
        
        classes = NEHRPClassification.EXTENDED_CLASSES if extended else NEHRPClassification.SITE_CLASSES
        
        for class_letter, bounds in classes.items():
            if bounds['vs30_min'] <= vs30 < bounds['vs30_max']:
                return class_letter, bounds['description']
        
        return 'Unknown', 'Classification failed'
    
    @staticmethod
    def get_site_coefficients(vs30, s_s=1.0, s_1=0.4):
        """
        Get site coefficients Fa and Fv for design spectrum
        
        Parameters:
        -----------
        vs30 : float
            Vs30 value (m/s)
        s_s : float
            Mapped MCE spectral acceleration at short periods
        s_1 : float
            Mapped MCE spectral acceleration at 1-second period
        
        Returns:
        --------
        fa : float
            Short-period site coefficient
        fv : float
            Long-period site coefficient
        """
        
        site_class, _ = NEHRPClassification.classify(vs30, extended=False)
        
        # Simplified coefficients (ASCE 7-16/22)
        # These are approximate; actual values depend on S_S and S_1
        
        fa_table = {
            'A': 0.8,
            'B': 1.0,
            'C': 1.2,
            'D': 1.6,
            'E': 2.5
        }
        
        fv_table = {
            'A': 0.8,
            'B': 1.0,
            'C': 1.7,
            'D': 2.4,
            'E': 3.5
        }
        
        fa = fa_table.get(site_class, 1.0)
        fv = fv_table.get(site_class, 1.0)
        
        return fa, fv


class EuroCode8Classification:
    """
    Eurocode 8 ground type classification
    """
    
    GROUND_TYPES = {
        'A': {'vs30_min': 800, 'vs30_max': np.inf, 'description': 'Rock'},
        'B': {'vs30_min': 360, 'vs30_max': 800, 'description': 'Very dense sand, gravel, or very stiff clay'},
        'C': {'vs30_min': 180, 'vs30_max': 360, 'description': 'Dense sand, gravel, or stiff clay'},
        'D': {'vs30_min': 100, 'vs30_max': 180, 'description': 'Loose to medium sand, gravel, or soft clay'},
        'E': {'vs30_min': 0, 'vs30_max': 100, 'description': 'Very soft clay or loose sand'},
    }
    
    @staticmethod
    def classify(vs30):
        """
        Classify site according to Eurocode 8
        
        Parameters:
        -----------
        vs30 : float
            Vs30 value (m/s)
        
        Returns:
        --------
        ground_type : str
            Eurocode 8 ground type
        description : str
            Ground type description
        """
        
        for ground_type, bounds in EuroCode8Classification.GROUND_TYPES.items():
            if bounds['vs30_min'] <= vs30 < bounds['vs30_max']:
                return ground_type, bounds['description']
        
        return 'Unknown', 'Classification failed'


def classify_site_all_systems(vs30):
    """
    Classify site according to multiple systems
    
    Parameters:
    -----------
    vs30 : float
        Vs30 value (m/s)
    
    Returns:
    --------
    classifications : dict
        Dictionary with classifications from different systems
    """
    
    nehrp_class, nehrp_desc = NEHRPClassification.classify(vs30, extended=False)
    nehrp_ext_class, nehrp_ext_desc = NEHRPClassification.classify(vs30, extended=True)
    ec8_class, ec8_desc = EuroCode8Classification.classify(vs30)
    fa, fv = NEHRPClassification.get_site_coefficients(vs30)
    
    classifications = {
        'vs30': vs30,
        'nehrp': {
            'class': nehrp_class,
            'description': nehrp_desc,
            'fa': fa,
            'fv': fv
        },
        'nehrp_extended': {
            'class': nehrp_ext_class,
            'description': nehrp_ext_desc
        },
        'eurocode8': {
            'class': ec8_class,
            'description': ec8_desc
        }
    }
    
    return classifications


if __name__ == "__main__":
    # Test classifications
    test_vs30_values = [150, 250, 400, 600, 900, 1200, 1800]
    
    print("=" * 70)
    print("SITE CLASSIFICATION TEST")
    print("=" * 70)
    
    for vs30 in test_vs30_values:
        print(f"\nVs30 = {vs30} m/s:")
        
        classifications = classify_site_all_systems(vs30)
        
        print(f"  NEHRP (5 classes): {classifications['nehrp']['class']} - {classifications['nehrp']['description']}")
        print(f"    Site coefficients: Fa = {classifications['nehrp']['fa']:.2f}, Fv = {classifications['nehrp']['fv']:.2f}")
        print(f"  NEHRP Extended (9 classes): {classifications['nehrp_extended']['class']} - {classifications['nehrp_extended']['description']}")
        print(f"  Eurocode 8: {classifications['eurocode8']['class']} - {classifications['eurocode8']['description']}")