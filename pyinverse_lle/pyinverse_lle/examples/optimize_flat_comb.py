"""
Main optimization script for flat comb generation.

Python equivalent of lle_dispersion_genetic_optimize.m
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyinverse_lle.utils.initialization import setup_poly_comb_target, validate_configuration
from pyinverse_lle.genetic.algorithm import GeneticOptimizer
from pyinverse_lle.utils.visualization import plot_dispersion, plot_spectrum
from pyinverse_lle.core.lle_solver import spectrum_f


def main():
    """Main optimization routine."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLE Dispersion Genetic Optimization')
    parser.add_argument('--testing', action='store_true', 
                       help='Run in testing mode (single generation, single individual)')
    parser.add_argument('--n_pop', type=int, default=208,
                       help='Population size (default: 208)')
    parser.add_argument('--max_iter', type=int, default=300,
                       help='Maximum iterations (default: 300)')
    parser.add_argument('--n_workers', type=int, default=-1,
                       help='Number of parallel workers (-1 for all cores)')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory (default: ./data)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    print("=== PyInverseLLE: Genetic Algorithm Optimization ===")
    print(f"Starting optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # ============================================================================
    # Setup optimization configuration
    # ============================================================================
    
    # System parameters
    n_modes = 512  # Number of modes (power of 2 for efficient FFT)
    
    # Setup for flat comb target (equivalent to init_polyCombTarget.m)
    config = setup_poly_comb_target(
        n_modes=n_modes,
        poly_order=6,
        f0_max=np.sqrt(20),
        target_power_db=None,  # Use MATLAB formula: -19 + target_var_power/2 = -16.5 dB
        target_var_power=5,  # Allowed variation in dB
        target_half_width_modes=32  # Half-width in modes (8 output optical ports)
    )
    
    # Validate configuration
    if not validate_configuration(config):
        print("Configuration validation failed!")
        return
    
    # ============================================================================
    # Initialize genetic algorithm
    # ============================================================================
    
    # GA parameters
    ga_params = {
        'n_pop': args.n_pop,
        'max_iter': args.max_iter,
        'fitness_tol': 1e-6,
        'mutation_proba': 0.15,
        'n_workers': args.n_workers if not args.testing else 1,
        'testing_mode': args.testing,
        'pulse_profile_init': config.get('pulse_profile_init')
    }
    
    # LLE parameters
    lle_params = {
        'lle_type': 1,  # Single direction
        'noise_factor': 0.0  # No noise for deterministic optimization
    }
    
    # Combine parameters
    ga_params.update(lle_params)
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        config['dispersion_param'],
        config['fitness_evaluator'],
        **ga_params
    )
    
    print(f"Configuration: {args.n_pop} individuals, {args.max_iter} max generations")
    print(f"Target: Flat comb, {config['parameters']['target_half_width_modes']} modes half-width")
    print(f"Target power: {config['parameters']['target_power_db']} ± {config['parameters']['target_var_power']/2} dB")
    
    # ============================================================================
    # Visualization of initial setup
    # ============================================================================
    
    if not args.no_plot:
        print("\nPlotting initial configuration...")
        
        # Plot initial dispersion profile
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        initial_dispersion = config['dispersion_param'].compute_dispersion(0)
        plot_dispersion(config['dispersion_param'].mu, initial_dispersion, 
                       title='Initial Dispersion Profile')
        
        plt.subplot(2, 2, 2)
        plot_spectrum(config['dispersion_param'].mu, config['comb_target'],
                     title='Target Comb Spectrum', ylabel='Power (linear)')
        
        plt.subplot(2, 2, 3)
        target_power_db = 10 * np.log10(config['comb_target'] + 1e-15)
        plt.stem(config['dispersion_param'].mu, 
                np.maximum(-100, target_power_db), 
                basefmt=' ', linefmt='r-', markerfmt='none')
        plt.xlim([-50, 50])
        plt.ylim([-100, 10])
        plt.xlabel('Mode #')
        plt.ylabel('Power (dB)')
        plt.title('Target Comb Spectrum (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
    
    # ============================================================================
    # Run optimization
    # ============================================================================
    
    print(f"\nStarting genetic algorithm optimization...")
    start_time = datetime.now()
    
    try:
        # Run optimization
        results = optimizer.optimize(
            save_data=True,
            output_dir=args.output_dir
        )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n=== Optimization completed in {total_time:.1f} seconds ===")
        print(f"Final generation: {results['final_generation']}")
        print(f"Best fitness: {results['best_fitness']:.6e}")
        
        # Get best individual
        best_idx, best_params = optimizer.get_best_individual()
        print(f"Best individual index: {best_idx}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        results = optimizer._compile_results()
        best_idx, best_params = optimizer.get_best_individual()
    except Exception as e:
        print(f"Optimization failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # Analyze and visualize results
    # ============================================================================
    
    if not args.no_plot:
        print("\nGenerating result plots...")
        
        # Final results figure
        plt.figure(figsize=(15, 10))
        
        # Fitness evolution
        plt.subplot(2, 3, 1)
        plt.semilogy(results['best_fitness_history'])
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Evolution')
        plt.grid(True, alpha=0.3)
        
        # Population diversity
        if len(results['diversity_history']) > 0:
            plt.subplot(2, 3, 2)
            plt.plot(results['diversity_history'])
            plt.xlabel('Generation')
            plt.ylabel('Population Diversity')
            plt.title('Population Diversity')
            plt.grid(True, alpha=0.3)
        
        # Best dispersion profile
        plt.subplot(2, 3, 3)
        best_dispersion = config['dispersion_param'].compute_dispersion(best_idx)
        plot_dispersion(config['dispersion_param'].mu, best_dispersion,
                       title='Optimized Dispersion Profile')
        
        # Best spectrum
        plt.subplot(2, 3, 4)
        best_psi = best_params['psi_final']
        if best_psi.ndim > 1:
            best_psi = best_psi[:, 0]  # Take first direction
        best_spectrum = spectrum_f(best_psi)
        
        plot_spectrum(config['dispersion_param'].mu, best_spectrum,
                     title='Optimized Comb Spectrum')
        
        # Spectrum comparison (dB)
        plt.subplot(2, 3, 5)
        target_db = 10 * np.log10(config['comb_target'] + 1e-15)
        result_db = 10 * np.log10(best_spectrum + 1e-15)
        
        plt.plot(config['dispersion_param'].mu, np.maximum(-100, target_db), 
                'r--', label='Target', linewidth=2)
        plt.plot(config['dispersion_param'].mu, np.maximum(-100, result_db), 
                'b-', label='Optimized', linewidth=2)
        
        plt.xlim([-50, 50])
        plt.ylim([-60, 10])
        plt.xlabel('Mode #')
        plt.ylabel('Power (dB)')
        plt.title('Spectrum Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fitness distribution
        plt.subplot(2, 3, 6)
        if results['fitness_evolution'].shape[1] > 0:
            final_fitness = results['fitness_evolution'][:, -1]
            plt.hist(final_fitness, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(results['best_fitness'], color='red', linestyle='--', 
                       label=f'Best: {results["best_fitness"]:.2e}')
            plt.xlabel('Fitness')
            plt.ylabel('Count')
            plt.title('Final Fitness Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No fitness data\n(testing mode)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Final Fitness Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ============================================================================
    # Print summary
    # ============================================================================
    
    print(f"\n=== Optimization Summary ===")
    print(f"Target: Flat comb with {config['parameters']['target_half_width_modes']} modes half-width")
    print(f"Population size: {args.n_pop}")
    print(f"Generations completed: {results['final_generation']}")
    print(f"Best fitness achieved: {results['best_fitness']:.6e}")
    print(f"Fitness tolerance: {ga_params['fitness_tol']:.6e}")
    
    convergence_status = "CONVERGED" if results['best_fitness'] < ga_params['fitness_tol'] else "NOT CONVERGED"
    print(f"Convergence status: {convergence_status}")
    
    # Best parameters summary
    print(f"\n=== Best Individual Parameters ===")
    print(f"Index: {best_idx}")
    for param_name, value in best_params.items():
        if param_name not in ['fitness', 'psi_final']:
            if isinstance(value, np.ndarray) and len(value) > 5:
                print(f"{param_name}: [{value[0]:.4f}, {value[1]:.4f}, ..., {value[-1]:.4f}] (length {len(value)})")
            else:
                print(f"{param_name}: {value}")
    
    print(f"\nOptimization completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Keep plots open if not in testing mode
    if '--testing' not in sys.argv and '--no_plot' not in sys.argv:
        plt.show()  # Keep plots open