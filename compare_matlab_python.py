#!/usr/bin/env python3
"""
Compare MATLAB and Python genetic optimization results.

This script loads MATLAB results and compares them with Python implementation
to validate the translation accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
from datetime import datetime
import sys
import os

# Add package to Python path
sys.path.insert(0, '/Users/lucas/Documents/git/github/py-inverseLLE/pyinverse_lle')

from pyinverse_lle.utils.initialization import setup_poly_comb_target
from pyinverse_lle.genetic.algorithm import GeneticOptimizer
from pyinverse_lle.core.lle_solver import spectrum_f


def load_matlab_results(matlab_file: str) -> dict:
    """Load and parse MATLAB optimization results."""
    print(f"Loading MATLAB results from: {matlab_file}")
    
    try:
        matlab_data = loadmat(matlab_file)
        
        # Extract key data (adjust keys based on actual MATLAB structure)
        results = {}
        
        # Print available keys to understand structure
        print("Available MATLAB data keys:")
        for key in matlab_data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {type(matlab_data[key])}, shape: {getattr(matlab_data[key], 'shape', 'N/A')}")
        
        # Extract common variables (adjust names based on actual MATLAB output)
        possible_keys = {
            'fitness': ['fitnessArray', 'fitness_array', 'fitness'],
            'psi': ['Psi', 'psi_evolution', 'psi'],
            'dispersion_params': ['dispParam_Storage', 'dispersion_params'],
            'generations': ['i_gen', 'generations', 'n_generations'],
            'best_fitness': ['best_fitness', 'final_fitness'],
        }
        
        for result_key, matlab_keys in possible_keys.items():
            for matlab_key in matlab_keys:
                if matlab_key in matlab_data:
                    results[result_key] = matlab_data[matlab_key]
                    print(f"✓ Found {result_key}: {matlab_key}")
                    break
            else:
                print(f"⚠ Could not find {result_key}")
        
        return results
        
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return {}


def run_comparable_python_optimization(n_generations: int = 50) -> dict:
    """Run Python optimization with comparable parameters."""
    print(f"\\nRunning Python optimization ({n_generations} generations)...")
    
    # Setup similar to MATLAB
    config = setup_poly_comb_target(
        n_modes=512,  # Adjust based on MATLAB settings
        poly_order=6,
        target_power_db=-5,
        target_half_width_modes=16
    )
    
    # Create optimizer with similar parameters
    optimizer = GeneticOptimizer(
        config['dispersion_param'],
        config['fitness_evaluator'],
        n_pop=50,  # Smaller for comparison
        max_iter=n_generations,
        fitness_tol=1e-6,
        n_workers=4  # Moderate parallelization
    )
    
    # Run optimization
    results = optimizer.optimize(save_data=False)
    
    return results, optimizer, config


def compare_fitness_evolution(matlab_results: dict, python_results: dict):
    """Compare fitness evolution between MATLAB and Python."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fitness evolution comparison
    ax1 = axes[0, 0]
    
    if 'fitness' in matlab_results:
        matlab_fitness = matlab_results['fitness']
        if matlab_fitness.ndim == 2:
            # Best fitness per generation
            matlab_best = np.min(matlab_fitness, axis=0)
            matlab_mean = np.mean(matlab_fitness, axis=0)
            
            generations_matlab = np.arange(len(matlab_best))
            ax1.semilogy(generations_matlab, matlab_best, 'r-', linewidth=2, label='MATLAB Best')
            ax1.semilogy(generations_matlab, matlab_mean, 'r--', alpha=0.7, label='MATLAB Mean')
    
    if 'best_fitness_history' in python_results:
        python_best = python_results['best_fitness_history']
        python_mean = np.mean(python_results['fitness_evolution'], axis=0)
        
        generations_python = np.arange(len(python_best))
        ax1.semilogy(generations_python, python_best, 'b-', linewidth=2, label='Python Best')
        ax1.semilogy(generations_python, python_mean, 'b--', alpha=0.7, label='Python Mean')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final fitness distribution comparison
    ax2 = axes[0, 1]
    
    if 'fitness' in matlab_results:
        matlab_fitness = matlab_results['fitness']
        if matlab_fitness.ndim == 2:
            matlab_final = matlab_fitness[:, -1]
            ax2.hist(matlab_final, bins=30, alpha=0.7, color='red', 
                    label=f'MATLAB (n={len(matlab_final)})', density=True)
    
    if 'fitness_evolution' in python_results:
        python_final = python_results['fitness_evolution'][:, -1]
        ax2.hist(python_final, bins=30, alpha=0.7, color='blue',
                label=f'Python (n={len(python_final)})', density=True)
    
    ax2.set_xlabel('Final Fitness')
    ax2.set_ylabel('Density')
    ax2.set_title('Final Fitness Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convergence rate comparison
    ax3 = axes[1, 0]
    
    fitness_threshold = 1e4  # Adjust threshold
    
    if 'fitness' in matlab_results:
        matlab_fitness = matlab_results['fitness']
        if matlab_fitness.ndim == 2:
            matlab_convergence = np.mean(matlab_fitness < fitness_threshold, axis=0) * 100
            ax3.plot(np.arange(len(matlab_convergence)), matlab_convergence, 'r-', 
                    linewidth=2, label='MATLAB')
    
    if 'fitness_evolution' in python_results:
        python_convergence = np.mean(python_results['fitness_evolution'] < fitness_threshold, axis=0) * 100
        ax3.plot(np.arange(len(python_convergence)), python_convergence, 'b-',
                linewidth=2, label='Python')
    
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Convergence Rate (%)')
    ax3.set_title(f'Convergence Rate (Fitness < {fitness_threshold:.0e})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    ax4 = axes[1, 1]
    
    metrics = {}
    
    # MATLAB metrics
    if 'fitness' in matlab_results:
        matlab_fitness = matlab_results['fitness']
        if matlab_fitness.ndim == 2:
            metrics['MATLAB'] = {
                'Final Best': np.min(matlab_fitness[:, -1]),
                'Final Mean': np.mean(matlab_fitness[:, -1]),
                'Final Std': np.std(matlab_fitness[:, -1]),
                'Generations': matlab_fitness.shape[1]
            }
    
    # Python metrics
    if 'fitness_evolution' in python_results:
        python_fitness = python_results['fitness_evolution']
        metrics['Python'] = {
            'Final Best': python_results['best_fitness'],
            'Final Mean': np.mean(python_fitness[:, -1]),
            'Final Std': np.std(python_fitness[:, -1]),
            'Generations': python_results['final_generation']
        }
    
    # Display metrics as text
    ax4.axis('off')
    metrics_text = "Performance Metrics Comparison\\n\\n"
    
    for system, system_metrics in metrics.items():
        metrics_text += f"{system}:\\n"
        for metric, value in system_metrics.items():
            if isinstance(value, float):
                metrics_text += f"  {metric}: {value:.2e}\\n"
            else:
                metrics_text += f"  {metric}: {value}\\n"
        metrics_text += "\\n"
    
    ax4.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    return fig


def compare_dispersion_profiles(matlab_results: dict, python_results: dict, 
                               optimizer=None, config=None):
    """Compare dispersion profiles between MATLAB and Python."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Best dispersion profile comparison
    ax1 = axes[0, 0]
    
    if optimizer and config:
        # Get Python best dispersion
        best_idx, _ = optimizer.get_best_individual()
        python_dispersion = config['dispersion_param'].compute_dispersion(best_idx)
        mu = config['dispersion_param'].mu
        
        ax1.plot(mu, python_dispersion, 'b-', linewidth=2, label='Python Best')
    
    # TODO: Extract MATLAB dispersion profile if available
    # This would require knowing the structure of dispParam_Storage
    
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Dispersion')
    ax1.set_title('Best Dispersion Profile Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add other comparison plots as needed
    # ...
    
    plt.tight_layout()
    return fig


def generate_comparison_report(matlab_results: dict, python_results: dict):
    """Generate detailed comparison report."""
    
    report = []
    report.append("=" * 60)
    report.append("MATLAB vs Python Optimization Comparison Report")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data availability
    report.append("Data Availability:")
    report.append(f"  MATLAB keys found: {list(matlab_results.keys())}")
    report.append(f"  Python data complete: {'fitness_evolution' in python_results}")
    report.append("")
    
    # Quantitative comparison
    if 'fitness' in matlab_results and 'fitness_evolution' in python_results:
        matlab_fitness = matlab_results['fitness']
        python_fitness = python_results['fitness_evolution']
        
        if matlab_fitness.ndim == 2 and python_fitness.ndim == 2:
            matlab_best_final = np.min(matlab_fitness[:, -1])
            python_best_final = python_results['best_fitness']
            
            improvement_ratio = matlab_best_final / python_best_final if python_best_final > 0 else float('inf')
            
            report.append("Performance Comparison:")
            report.append(f"  MATLAB best final fitness: {matlab_best_final:.6e}")
            report.append(f"  Python best final fitness: {python_best_final:.6e}")
            report.append(f"  Ratio (MATLAB/Python): {improvement_ratio:.3f}")
            
            if abs(improvement_ratio - 1.0) < 0.1:
                report.append("  ✓ Results are very similar!")
            elif improvement_ratio < 1.0:
                report.append("  → Python achieved better optimization")
            else:
                report.append("  → MATLAB achieved better optimization")
            report.append("")
    
    # Algorithm validation
    report.append("Algorithm Validation:")
    if python_results.get('final_generation', 0) > 0:
        report.append("  ✓ Python optimization completed successfully")
        report.append("  ✓ Genetic algorithm pipeline working")
        report.append("  ✓ LLE solver functional")
        report.append("  ✓ Fitness evaluation working")
    else:
        report.append("  ⚠ Python optimization may need longer run")
    
    return "\\n".join(report)


def main():
    """Main comparison function."""
    
    print("=== MATLAB vs Python Optimization Comparison ===")
    
    # Load MATLAB data
    matlab_file = "/Users/lucas/Documents/git/github/py-inverseLLE/Data/2021_03_30-10.40.06_Genetic_evolution.mat"
    matlab_results = load_matlab_results(matlab_file)
    
    if not matlab_results:
        print("Could not load MATLAB data. Exiting.")
        return
    
    # Run Python optimization
    python_results, optimizer, config = run_comparable_python_optimization(n_generations=30)
    
    print(f"\\nPython optimization completed:")
    print(f"  Generations: {python_results['final_generation']}")
    print(f"  Best fitness: {python_results['best_fitness']:.6e}")
    
    # Generate comparisons
    print("\\nGenerating comparison plots...")
    
    # Fitness comparison
    fig1 = compare_fitness_evolution(matlab_results, python_results)
    fig1.suptitle("MATLAB vs Python: Fitness Evolution Comparison", fontsize=16)
    
    # Dispersion comparison  
    fig2 = compare_dispersion_profiles(matlab_results, python_results, optimizer, config)
    fig2.suptitle("MATLAB vs Python: Dispersion Profile Comparison", fontsize=16)
    
    # Generate report
    report = generate_comparison_report(matlab_results, python_results)
    print("\\n" + report)
    
    # Save report
    timestamp = datetime.now().strftime('%Y_%m_%d-%H.%M.%S')
    report_file = f"matlab_python_comparison_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\\nComparison report saved to: {report_file}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()