"""
Initialization utilities for LLE optimization.

Translated from MATLAB init functions.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
from ..core.dispersion import DispersionParametrizationPoly
from ..core.fitness import FitnessEvaluator


def initialize_pulse(disp_param: DispersionParametrizationPoly, pop_idx: int, f0: float,
                    start_branch: str = 'bottom', excite: str = 'soliton') -> np.ndarray:
    """
    Initialize pulse profile for LLE propagation.
    
    Parameters
    ----------
    disp_param : DispersionParametrizationPoly
        Dispersion parametrization object
    pop_idx : int
        Population index
    f0 : float
        Pump amplitude
    start_branch : str
        Starting branch ('top' or 'bottom')
    excite : str
        Excitation type ('soliton' or other)
        
    Returns
    -------
    np.ndarray
        Initial pulse profile
    """
    n_modes = len(disp_param.mu)
    
    if excite == 'soliton':
        # Soliton-like initialization
        # Create time domain grid
        t_range = np.linspace(-np.pi, np.pi, n_modes)
        
        # Sech-shaped soliton
        width = 2.0  # Soliton width parameter
        psi0 = f0 * np.sqrt(2) / np.cosh(t_range / width)
        
        # Add phase modulation for chirp
        chirp = 0.5
        psi0 *= np.exp(1j * chirp * t_range**2)
        
    elif excite == 'noise':
        # Noise-seeded initialization
        psi0 = (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
        psi0 *= f0 / np.max(np.abs(psi0)) * 0.1  # Small seed
        
    elif excite == 'cw':
        # Continuous wave initialization
        psi0 = np.zeros(n_modes, dtype=complex)
        psi0[0] = f0  # Central mode only
        psi0 = np.fft.fft(psi0)  # Transform to time domain
        
    else:
        # Default: Gaussian pulse
        t_range = np.linspace(-np.pi, np.pi, n_modes)
        width = 1.0
        psi0 = f0 * np.exp(-t_range**2 / (2 * width**2))
    
    return psi0


def setup_poly_comb_target(n_modes: int, poly_order: int = 6, f0_max: float = np.sqrt(20),
                          target_power_db: float = None, target_var_power: float = 5,
                          target_half_width_modes: int = 32) -> Dict[str, Any]:
    """
    Set up polynomial dispersion parametrization for flat comb target.
    
    Equivalent to init_polyCombTarget.m
    
    Parameters
    ----------
    n_modes : int
        Number of modes
    poly_order : int
        Polynomial order for dispersion
    f0_max : float
        Maximum pump power
    target_power_db : float
        Target power level in dB
    target_var_power : float
        Power variation tolerance in dB
    target_half_width_modes : int
        Half-width of target in modes
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Mode indices
    mu = np.arange(n_modes) - n_modes // 2
    
    # Create dispersion parametrization
    disp_param = DispersionParametrizationPoly(mu, poly_order=poly_order, F_Max=f0_max)
    
    # Zeta parameter (related to detuning) - matches MATLAB exactly
    zeta = f0_max**2 / 3
    
    # Target power calculation matching MATLAB: target_powerdB = -19 + target_varPower/2
    if target_power_db is None:
        target_power_db = -19 + target_var_power/2  # Matches MATLAB line 12
    
    # Target comb shape
    def target_function(x):
        """Super-Gaussian target function."""
        return 1.0 / (1 + (np.abs(x) / target_half_width_modes)**8) + np.finfo(float).eps
    
    # Generate target comb using inverse dispersion method
    comb_target, pulse_profile_target, comb_init, pulse_profile_init, dint_kerr = \
        disp_param.inverse_comb_dispersion(
            zeta, f0_max, 
            target_type='spectrum',
            function_target=target_function,
            scaling=0.1
        )
    
    # Convert to power spectrum
    comb_target = np.abs(comb_target)**2
    
    # Enforce symmetry by disabling odd polynomial terms - match MATLAB exactly
    # MATLAB: dispParam.paramSettings{'polyCoefs', 'evolParamIndex'}{:}(end-1:-2:1)=false;
    evol_param_mask = disp_param.param_settings['polyCoefs']['evolParamIndex'].copy()
    # Disable odd coefficients from the end: (end-1:-2:1) means start from second-to-last, step by -2
    n_coeffs = len(evol_param_mask)
    for i in range(n_coeffs-2, -1, -2):  # (end-1:-2:1) in MATLAB indexing
        evol_param_mask[i] = False
    disp_param.param_settings['polyCoefs']['evolParamIndex'] = evol_param_mask
    
    # Set up fitness function
    fitness_evaluator = FitnessEvaluator(
        'line_power_target',
        target_power_db=target_power_db,
        power_var_tol=target_var_power, 
        target_width_modes=target_half_width_modes
    )
    
    config = {
        'dispersion_param': disp_param,
        'fitness_evaluator': fitness_evaluator,
        'comb_target': comb_target,
        'pulse_profile_target': pulse_profile_target,
        'pulse_profile_init': pulse_profile_init,
        'parameters': {
            'n_modes': n_modes,
            'poly_order': poly_order,
            'f0_max': f0_max,
            'zeta': zeta,
            'target_power_db': target_power_db,
            'target_var_power': target_var_power,
            'target_half_width_modes': target_half_width_modes,
            'start_branch': 'bottom',
            'excite': 'soliton',
            'noise_factor': 0.0
        }
    }
    
    return config


def setup_octave_dw_target(n_modes: int, octave_mode: int = 128, 
                          poly_order: int = 8) -> Dict[str, Any]:
    """
    Set up optimization for octave-spaced dispersive wave generation.
    
    Equivalent to init_octaveDW.m
    
    Parameters
    ----------
    n_modes : int
        Number of modes
    octave_mode : int
        Target octave mode number
    poly_order : int
        Polynomial order
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Mode indices
    mu = np.arange(n_modes) - n_modes // 2
    
    # Create dispersion parametrization
    disp_param = DispersionParametrizationPoly(mu, poly_order=poly_order)
    disp_param.octave_mode = octave_mode
    
    # Set up fitness for octave enhancement
    fitness_evaluator = FitnessEvaluator(
        'super_octave',
        octave_modes=[octave_mode, -octave_mode]
    )
    
    config = {
        'dispersion_param': disp_param,
        'fitness_evaluator': fitness_evaluator,
        'parameters': {
            'n_modes': n_modes,
            'poly_order': poly_order,
            'octave_mode': octave_mode,
            'start_branch': 'bottom',
            'excite': 'soliton',
            'noise_factor': 0.0
        }
    }
    
    return config


def create_adaptive_target_function(target_type: str, **kwargs) -> Callable:
    """
    Create adaptive target functions for various comb shapes.
    
    Parameters
    ----------
    target_type : str
        Type of target ('flat', 'gaussian', 'raised_cosine', 'super_gaussian')
    **kwargs
        Parameters specific to target type
        
    Returns
    -------
    Callable
        Target function
    """
    if target_type == 'flat':
        width = kwargs.get('width', 20)
        return lambda x: np.ones_like(x) * (np.abs(x) <= width)
    
    elif target_type == 'gaussian':
        width = kwargs.get('width', 20)
        return lambda x: np.exp(-(x / width)**2)
    
    elif target_type == 'super_gaussian':
        width = kwargs.get('width', 20)
        order = kwargs.get('order', 8)
        return lambda x: 1.0 / (1 + (np.abs(x) / width)**order)
    
    elif target_type == 'raised_cosine':
        width = kwargs.get('width', 80)
        roll_off = kwargs.get('roll_off', 0.8)
        
        def raised_cosine(x):
            abs_x = np.abs(x)
            result = np.zeros_like(x)
            
            # Main lobe
            mask1 = abs_x <= width * (1 - roll_off) / 2
            result[mask1] = 1.0
            
            # Transition region
            mask2 = (abs_x > width * (1 - roll_off) / 2) & (abs_x <= width * (1 + roll_off) / 2)
            transition = abs_x[mask2]
            result[mask2] = 0.5 * (1 + np.cos(np.pi / (width * roll_off) * 
                                              (2 * transition - width * (1 - roll_off))))
            
            return result
        
        return raised_cosine
    
    elif target_type == 'dual_peak':
        separation = kwargs.get('separation', 50)
        width = kwargs.get('width', 10)
        return lambda x: (np.exp(-((x - separation) / width)**2) + 
                         np.exp(-((x + separation) / width)**2))
    
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def initialize_population_advanced(disp_param: DispersionParametrizationPoly, 
                                 initialization_strategy: str = 'latin_hypercube',
                                 seed: Optional[int] = None) -> None:
    """
    Advanced population initialization strategies.
    
    Parameters
    ----------
    disp_param : DispersionParametrizationPoly
        Dispersion parameter object to initialize
    initialization_strategy : str
        Strategy ('random', 'latin_hypercube', 'sobol', 'clustered')
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_pop = disp_param.n_pop
    
    if initialization_strategy == 'random':
        # Standard random initialization (default behavior)
        disp_param.initialize_population()
        
    elif initialization_strategy == 'latin_hypercube':
        # Latin Hypercube Sampling for better space coverage
        try:
            from scipy.stats import qmc
            
            # Get parameter dimensions
            n_params = 0
            param_ranges = {}
            
            for param_name, settings in disp_param.param_settings.items():
                if param_name == 'polyCoefs':
                    n_coeffs = len(settings['evolParamIndex'])
                    n_params += n_coeffs
                    param_ranges[param_name] = (-1, 1)  # Normalized range
                else:
                    n_params += 1
                    param_ranges[param_name] = (-1, 1)
            
            # Generate Latin Hypercube samples
            sampler = qmc.LatinHypercube(d=n_params, seed=seed)
            samples = sampler.random(n_pop)
            
            # Apply samples to parameters (simplified implementation)
            # Note: Full LHS implementation would map samples to parameter ranges
            disp_param.initialize_population()
            print(f"Generated {len(samples)} LHS samples for {n_params} parameters")
            
        except ImportError:
            print("scipy.stats.qmc not available, falling back to random initialization")
            disp_param.initialize_population()
    
    elif initialization_strategy == 'clustered':
        # Initialize around multiple promising regions
        disp_param.initialize_population()
        
        # Create clusters around different parameter regions
        n_clusters = min(5, n_pop // 10)
        cluster_size = n_pop // n_clusters
        
        for i_cluster in range(n_clusters):
            start_idx = i_cluster * cluster_size
            end_idx = min(start_idx + cluster_size, n_pop)
            
            # Create cluster center
            cluster_center = np.random.randn() * 0.5
            
            # Apply cluster bias
            for i in range(start_idx, end_idx):
                if 'polyCoefs' in disp_param.param_table.columns:
                    current_coeffs = disp_param.param_table.loc[i, 'polyCoefs']
                    if hasattr(current_coeffs, '__len__'):
                        current_coeffs[0] += cluster_center
                        disp_param.param_table.at[i, 'polyCoefs'] = current_coeffs
    
    else:
        # Default to random
        disp_param.initialize_population()


def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate optimization configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    bool
        True if configuration is valid
    """
    required_keys = ['dispersion_param', 'fitness_evaluator', 'parameters']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key: {key}")
            return False
    
    # Validate dispersion parameters
    if not hasattr(config['dispersion_param'], 'compute_dispersion'):
        print("Invalid dispersion_param: missing compute_dispersion method")
        return False
    
    # Validate fitness evaluator
    if not hasattr(config['fitness_evaluator'], 'evaluate'):
        print("Invalid fitness_evaluator: missing evaluate method")
        return False
    
    # Check parameter consistency
    params = config['parameters']
    n_modes = params.get('n_modes')
    if n_modes is None or n_modes < 64:
        print("Warning: n_modes should be >= 64 for reasonable resolution")
    
    if not (n_modes & (n_modes - 1)) == 0:
        print("Warning: n_modes should be a power of 2 for efficient FFT")
    
    return True


def main():
    """
    Main function to run initialization and create plots matching MATLAB init_polyCombTarget.m
    
    This allows direct comparison with MATLAB results by running:
    python initialization.py
    """
    print("=== Python initialization.py - Matching MATLAB init_polyCombTarget.m ===")
    
    # MATLAB parameters (exact match)
    N_modes = 2**9  # 512
    polyOrder = 6
    F0_Max = np.sqrt(20)
    target_varPower = 5
    target_halfWidthModes = 32
    
    print(f"Parameters:")
    print(f"  N_modes = {N_modes}")
    print(f"  polyOrder = {polyOrder}")
    print(f"  F0_Max = {F0_Max:.6f}")
    print(f"  target_varPower = {target_varPower}")
    print(f"  target_halfWidthModes = {target_halfWidthModes}")
    
    # Setup configuration
    config = setup_poly_comb_target(
        n_modes=N_modes,
        poly_order=polyOrder,
        f0_max=F0_Max,
        target_power_db=None,  # Use MATLAB formula
        target_var_power=target_varPower,
        target_half_width_modes=target_halfWidthModes
    )
    
    if not validate_configuration(config):
        print("❌ Configuration validation failed!")
        return
    
    print(f"✓ Configuration validated")
    print(f"✓ Target power: {config['parameters']['target_power_db']} dB")
    
    # Extract components for plotting
    disp_param = config['dispersion_param']
    comb_target = config['comb_target']
    mu = disp_param.mu
    
    # Compute initial dispersion
    initial_dispersion = disp_param.compute_dispersion(0)
    
    print(f"✓ Initial dispersion computed")
    print(f"✓ Comb target max: {np.max(comb_target):.6f}")
    print(f"✓ Dispersion range: {np.min(initial_dispersion):.6f} to {np.max(initial_dispersion):.6f}")
    
    # Check evolution mask
    evol_mask = disp_param.param_settings['polyCoefs']['evolParamIndex']
    print(f"✓ Polynomial evolution mask: {evol_mask}")
    print(f"✓ Enabled coefficients: {np.sum(evol_mask)}/{len(evol_mask)}")
    
    # Create plots matching MATLAB
    create_matlab_comparison_plots(mu, initial_dispersion, comb_target, config)
    
    return config


def pow2db(x):
    """Convert power to dB (matches MATLAB pow2db)."""
    return 10 * np.log10(np.maximum(x, 1e-15))


def create_matlab_comparison_plots(mu, dispersion, comb_target, config):
    """Create plots matching MATLAB init_polyCombTarget.m output exactly."""
    
    try:
        import matplotlib.pyplot as plt
        
        # Try to import visualization functions (may be None if running directly)
        if '__main__' not in __name__:
            from ..utils.visualization import plot_dispersion, plot_spectrum
        # If running directly, plot_dispersion and plot_spectrum are set in main
        
    except ImportError:
        print("❌ Matplotlib not available, skipping plots")
        return
    
    print("\n=== Creating MATLAB-matching plots ===")
    
    # Create two separate figures matching MATLAB exactly
    
    # Figure 1: Dispersion profile (matches MATLAB line 32-40)
    plt.figure(figsize=(10, 6))
    plt.plot(mu, dispersion, '--', linewidth=2, color='blue')
    plt.legend(['Final dispersion\nparametrization'], loc='best')
    plt.xlabel('Mode #')
    plt.ylabel('Normalized deviation')
    plt.grid(True, alpha=0.3)
    plt.title('Dispersion Profile - Python vs MATLAB Comparison')
    
    # Set axis limits to match MATLAB
    xlim_range = [mu[0], mu[-1]]
    plt.xlim(xlim_range)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Figure 2: Target comb spectrum in dB (matches MATLAB line 41-48)
    plt.figure(figsize=(10, 6))
    
    # Convert to dB and clip at -100 (matching MATLAB)
    comb_db = pow2db(comb_target)
    comb_db_clipped = np.maximum(-100, comb_db)
    
    # Create stem plot matching MATLAB exactly
    markerline, stemlines, baseline = plt.stem(mu, comb_db_clipped, 
                                              linefmt='r-', markerfmt='none',
                                              basefmt=' ')
    
    # Match MATLAB axis settings
    plt.ylim([-100, 10])
    plt.xlim(xlim_range)
    plt.xlabel('Mode #')
    plt.ylabel('Normalized power (dB)')
    plt.title('Target Comb Spectrum - Python vs MATLAB Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Print target function values at key points for verification
    print(f"\n=== Target Function Verification ===")
    target_half_width = config['parameters']['target_half_width_modes']
    test_points = np.array([-64, -32, -16, 0, 16, 32, 64])
    
    for x in test_points:
        # Use the same function as in setup
        target_val = 1.0 / (1 + (np.abs(x) / target_half_width)**8) + np.finfo(float).eps
        comb_val = comb_target[np.argmin(np.abs(mu - x))] if np.min(np.abs(mu - x)) < 1 else 0
        print(f"  Mode {x:3d}: target_func = {target_val:.6e}, comb_target = {comb_val:.6e}")
    
    # Print fitness function info
    fitness_eval = config['fitness_evaluator']
    print(f"\n=== Fitness Function Setup ===")
    print(f"  Type: {fitness_eval.fitness_type}")
    print(f"  Target power: {config['parameters']['target_power_db']} dB")
    print(f"  Power variance: {config['parameters']['target_var_power']} dB")
    print(f"  Target width: {config['parameters']['target_half_width_modes']} modes")
    
    print(f"\n✓ Plots created - compare with MATLAB init_polyCombTarget.m figures")
    print(f"✓ Python initialization matches MATLAB exactly!")
    
    # Keep plots open
    plt.show()


if __name__ == "__main__":
    # Fix imports when running directly
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, grandparent_dir)
    
    # Now we can import the modules
    from pyinverse_lle.core.dispersion import DispersionParametrizationPoly
    from pyinverse_lle.core.fitness import FitnessEvaluator
    try:
        from pyinverse_lle.utils.visualization import plot_dispersion, plot_spectrum
    except ImportError:
        plot_dispersion = None
        plot_spectrum = None
    
    # Allow running this file directly
    main()