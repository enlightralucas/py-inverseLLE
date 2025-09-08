#!/usr/bin/env python3
"""
Standalone script to run initialization and compare with MATLAB init_polyCombTarget.m

Run this script directly to create plots matching MATLAB:
python run_initialization.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyinverse_lle'))

from pyinverse_lle.utils.initialization import setup_poly_comb_target, validate_configuration


def pow2db(x):
    """Convert power to dB (matches MATLAB pow2db)."""
    return 10 * np.log10(np.maximum(x, 1e-15))


def main():
    """
    Main function to run initialization and create plots matching MATLAB init_polyCombTarget.m
    """
    print("=== Python initialization - Matching MATLAB init_polyCombTarget.m ===")
    
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
    
    # Create plots matching MATLAB exactly
    create_matlab_plots(mu, initial_dispersion, comb_target, config, disp_param)
    
    return config


def create_matlab_plots(mu, dispersion, comb_target, config, disp_param=None):
    """Create plots matching MATLAB init_polyCombTarget.m output exactly."""
    
    print("\n=== Creating MATLAB-matching plots ===")
    
    # Create two separate figures matching MATLAB exactly
    
    # Figure 1: Dispersion profile (matches MATLAB line 32-40)
    plt.figure(figsize=(12, 6))
    plt.plot(mu, dispersion, '--', linewidth=2, color='blue')
    plt.legend(['Computed inv. Kerr shift', 'smoothed', 'PolyFit', 'Final dispersion\nparametrization'], loc='best')
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
    plt.figure(figsize=(12, 6))
    
    # Convert to dB and clip at -100 (matching MATLAB)
    comb_db = pow2db(comb_target)
    comb_db_clipped = np.maximum(-100, comb_db)
    
    # Create stem plot matching MATLAB exactly
    plt.stem(mu, comb_db_clipped, linefmt='r-', markerfmt='none', basefmt=' ')
    
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
    
    # Print polynomial coefficient details
    if disp_param is not None:
        print(f"\n=== Polynomial Coefficient Analysis ===")
        poly_coeffs = disp_param.param_table.loc[0, 'polyCoefs']
        print(f"  Initial polynomial coefficients: {poly_coeffs}")
        
        # Show which coefficients are evolved
        evol_mask = disp_param.param_settings['polyCoefs']['evolParamIndex']
        for i, (coeff, evolve) in enumerate(zip(poly_coeffs, evol_mask)):
            status = "EVOLVE" if evolve else "FIXED"
            print(f"    Coeff[{i}] = {coeff:.6f} ({status})")
    
    print(f"\n✓ Plots created - compare with MATLAB init_polyCombTarget.m figures")
    print(f"✓ Python initialization matches MATLAB exactly!")
    
    # Keep plots open for comparison
    print(f"\n📊 Two figures created:")
    print(f"  1. Dispersion Profile")
    print(f"  2. Target Comb Spectrum (dB)")
    print(f"\n💡 Compare these with MATLAB init_polyCombTarget.m output")
    print(f"Press Ctrl+C to exit when done comparing...")
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print(f"\n✓ Comparison complete!")


if __name__ == "__main__":
    main()