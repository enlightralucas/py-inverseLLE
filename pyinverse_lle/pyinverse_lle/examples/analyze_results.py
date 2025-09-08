"""
Analysis script for genetic optimization results.

Python equivalent of analyze_genetic_dispersion_results.m
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyinverse_lle.utils.data_io import load_results, export_summary_report, export_best_parameters_csv
from pyinverse_lle.utils.visualization import (plot_evolution_summary, plot_comparison_with_target,
                                              plot_spectrum, plot_dispersion, plot_stem_spectrum)
from pyinverse_lle.core.lle_solver import spectrum_f, LLEPropagator
from pyinverse_lle.core.dispersion import DispersionParametrizationPoly
from pyinverse_lle.utils.initialization import setup_poly_comb_target


def analyze_optimization_results(results_file: str, config_params: dict = None, 
                                save_plots: bool = True, show_plots: bool = True):
    """
    Analyze optimization results from saved file.
    
    Parameters
    ----------
    results_file : str
        Path to results file
    config_params : dict, optional
        Configuration parameters (if not in results file)
    save_plots : bool
        Whether to save plots
    show_plots : bool
        Whether to show plots interactively
    """
    print(f"Loading results from: {results_file}")
    
    # Load results
    results = load_results(results_file)
    
    # Extract key information
    n_generations = results.get('final_generation', 0)
    best_fitness = results.get('best_fitness', np.inf)
    best_idx = results.get('best_individual_idx', 0)
    
    print(f"Optimization completed after {n_generations} generations")
    print(f"Best fitness achieved: {best_fitness:.6e}")
    print(f"Best individual index: {best_idx}")
    
    # Get optimization parameters
    opt_params = results.get('optimization_params', {})
    n_modes = opt_params.get('n_modes', 512)
    fitness_tol = opt_params.get('fitness_tol', 1e-6)
    
    print(f"Convergence status: {'CONVERGED' if best_fitness < fitness_tol else 'NOT CONVERGED'}")
    
    # ========================================================================
    # Reconstruct dispersion parametrization and target
    # ========================================================================
    
    # Try to reconstruct from results or use provided config
    if config_params is None:
        # Use default parameters similar to the optimization
        config = setup_poly_comb_target(n_modes=n_modes)
    else:
        config = config_params
    
    disp_param = config['dispersion_param']
    comb_target = config['comb_target']
    
    # Update parameters with best individual
    if 'dispersion_params' in results and len(results['dispersion_params']) > 0:
        final_params = results['dispersion_params'][-1]
        best_params_row = final_params.iloc[best_idx]
        
        # Update dispersion parameters
        for param_name in disp_param.param_table.columns:
            if param_name in best_params_row.index:
                disp_param.param_table.loc[0, param_name] = best_params_row[param_name]
    
    # ========================================================================
    # Recompute best solution
    # ========================================================================
    
    print("Recomputing best solution...")
    
    # Get best field from results if available
    if 'psi_evolution' in results:
        best_psi = results['psi_evolution'][:, :, best_idx, -1]
        if best_psi.ndim > 1:
            best_psi = best_psi[:, 0]  # Take first direction
    else:
        # Recompute by running LLE
        print("Field data not in results, recomputing...")
        
        # Create LLE propagator
        lle_propagator = LLEPropagator(n_modes)
        
        # Get parameters
        best_dispersion = disp_param.compute_dispersion(0)
        dispersion = -1.0 - 1j * np.fft.ifftshift(best_dispersion)
        
        zeta = disp_param.param_table.loc[0, 'detuning'] if 'detuning' in disp_param.param_table.columns else 0
        f0 = disp_param.param_table.loc[0, 'pumpPow'] if 'pumpPow' in disp_param.param_table.columns else np.sqrt(20)
        
        # Initialize pulse (simplified)
        t_range = np.linspace(-np.pi, np.pi, n_modes)
        psi0 = f0 * np.sqrt(2) / np.cosh(t_range / 2.0)
        
        # Propagate
        best_psi, _, _ = lle_propagator.propagate(
            psi0, f0, zeta, dispersion, 0.0,  # No noise
            40.0, 2**(-9), 1000
        )
    
    # Compute spectrum
    best_spectrum = spectrum_f(best_psi)
    
    # ========================================================================
    # Generate comprehensive analysis plots
    # ========================================================================
    
    if save_plots or show_plots:
        output_dir = os.path.dirname(results_file)
        base_name = os.path.splitext(os.path.basename(results_file))[0]
        
        print("Generating analysis plots...")
        
        # 1. Evolution summary
        fig1 = plot_evolution_summary(results, 
                                     save_path=os.path.join(output_dir, f'{base_name}_evolution.png') if save_plots else None)
        
        # 2. Best solution analysis
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dispersion profile
        best_dispersion = disp_param.compute_dispersion(0)
        plot_dispersion(disp_param.mu, best_dispersion, 
                       title='Optimized Dispersion Profile', ax=ax1)
        
        # Spectrum comparison
        plot_spectrum(disp_param.mu, best_spectrum, 
                     title='Optimized Spectrum', ax=ax2)
        ax2.plot(disp_param.mu, comb_target, 'r--', alpha=0.7, label='Target')
        ax2.legend()
        
        # dB comparison
        target_db = 10 * np.log10(np.maximum(comb_target, 1e-15))
        result_db = 10 * np.log10(np.maximum(best_spectrum, 1e-15))
        
        ax3.plot(disp_param.mu, target_db, 'r--', linewidth=2, label='Target')
        ax3.plot(disp_param.mu, result_db, 'b-', linewidth=2, label='Optimized')
        ax3.set_xlim([-50, 50])
        ax3.set_ylim([-60, 10])
        ax3.set_xlabel('Mode #')
        ax3.set_ylabel('Power (dB)')
        ax3.set_title('Spectrum Comparison (dB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Time domain pulse
        time_pulse = np.fft.ifft(best_psi)
        time_axis = np.linspace(-np.pi, np.pi, len(time_pulse))
        
        ax4.plot(time_axis, np.abs(time_pulse)**2, 'b-', linewidth=2)
        ax4.set_xlabel('Time (normalized)')
        ax4.set_ylabel('|ψ|² (Power)')
        ax4.set_title('Temporal Pulse Profile')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, f'{base_name}_best_solution.png'), 
                       dpi=300, bbox_inches='tight')
        
        # 3. Detailed spectrum comparison
        fig3 = plot_comparison_with_target(
            disp_param.mu, comb_target, best_spectrum,
            title='Target vs Optimized Spectrum Detailed Comparison',
            xlim=[-50, 50],
            save_path=os.path.join(output_dir, f'{base_name}_comparison.png') if save_plots else None
        )
        
        # 4. Parameter evolution (if available)
        if 'dispersion_params' in results and len(results['dispersion_params']) > 1:
            fig4, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            n_gens = len(results['dispersion_params'])
            generations = np.arange(n_gens)
            
            # Track specific parameters
            param_names = ['detuning', 'pumpPow']
            if 'polyCoefs' in results['dispersion_params'][0].columns:
                param_names.extend(['poly_coeff_0', 'poly_coeff_1'])  # First two coefficients
            
            for i, param_name in enumerate(param_names[:4]):
                if i < len(axes):
                    ax = axes[i]
                    
                    if param_name.startswith('poly_coeff'):
                        coeff_idx = int(param_name.split('_')[-1])
                        param_evolution = []
                        
                        for gen in range(n_gens):
                            best_gen_idx = np.argmin(results['fitness_evolution'][:, gen])
                            poly_coeffs = results['dispersion_params'][gen].loc[best_gen_idx, 'polyCoefs']
                            if hasattr(poly_coeffs, '__len__') and len(poly_coeffs) > coeff_idx:
                                param_evolution.append(poly_coeffs[coeff_idx])
                            else:
                                param_evolution.append(0)
                        
                        ax.plot(generations, param_evolution, 'b-', linewidth=2)
                        ax.set_title(f'Best Individual: Polynomial Coefficient {coeff_idx}')
                    else:
                        if param_name in results['dispersion_params'][0].columns:
                            param_evolution = []
                            
                            for gen in range(n_gens):
                                best_gen_idx = np.argmin(results['fitness_evolution'][:, gen])
                                param_evolution.append(results['dispersion_params'][gen].loc[best_gen_idx, param_name])
                            
                            ax.plot(generations, param_evolution, 'g-', linewidth=2)
                            ax.set_title(f'Best Individual: {param_name}')
                    
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Parameter Value')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(os.path.join(output_dir, f'{base_name}_parameter_evolution.png'), 
                           dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        
        print(f"Analysis plots completed.")
        if save_plots:
            print(f"Plots saved to: {output_dir}")
    
    # ========================================================================
    # Generate summary reports
    # ========================================================================
    
    if save_plots:  # Use same flag for saving reports
        print("Generating summary reports...")
        
        # Text summary
        summary_file = os.path.join(output_dir, f'{base_name}_summary.txt')
        export_summary_report(results, summary_file)
        
        # CSV export of best parameters
        csv_file = os.path.join(output_dir, f'{base_name}_best_parameters.csv')
        export_best_parameters_csv(results, csv_file)
    
    # ========================================================================
    # Performance metrics
    # ========================================================================
    
    print("\n=== Performance Analysis ===")
    
    # Fitness improvement
    if 'best_fitness_history' in results:
        initial_fitness = results['best_fitness_history'][0]
        final_fitness = results['best_fitness_history'][-1]
        improvement = initial_fitness / final_fitness
        print(f"Fitness improvement: {improvement:.2f}x ({initial_fitness:.2e} → {final_fitness:.2e})")
    
    # Convergence analysis
    if 'fitness_evolution' in results:
        final_gen_fitness = results['fitness_evolution'][:, -1]
        converged_count = np.sum(final_gen_fitness < fitness_tol)
        convergence_rate = converged_count / len(final_gen_fitness) * 100
        print(f"Final convergence rate: {convergence_rate:.1f}% ({converged_count}/{len(final_gen_fitness)})")
        
        # Statistics
        print(f"Final population fitness - Mean: {np.mean(final_gen_fitness):.2e}, "
              f"Std: {np.std(final_gen_fitness):.2e}")
    
    # Spectral metrics
    roi_mask = np.abs(disp_param.mu) <= 20  # Region of interest
    target_power_roi = np.sum(comb_target[roi_mask])
    result_power_roi = np.sum(best_spectrum[roi_mask])
    power_ratio = result_power_roi / target_power_roi
    
    print(f"Power in ROI - Target: {target_power_roi:.4f}, Result: {result_power_roi:.4f}, "
          f"Ratio: {power_ratio:.3f}")
    
    # Spectral fidelity (correlation coefficient)
    correlation = np.corrcoef(comb_target[roi_mask], best_spectrum[roi_mask])[0, 1]
    print(f"Spectral correlation (ROI): {correlation:.4f}")
    
    print(f"\nAnalysis completed successfully!")
    
    return results, best_spectrum, disp_param


def main():
    """Main analysis function.""" 
    parser = argparse.ArgumentParser(description='Analyze LLE Genetic Optimization Results')
    parser.add_argument('results_file', type=str, 
                       help='Path to results file (.npz, .h5, .pkl)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save analysis plots and reports')
    parser.add_argument('--no_show', action='store_true', 
                       help='Do not show plots interactively')
    parser.add_argument('--find_latest', action='store_true',
                       help='Find and analyze the latest results file in ./data/')
    
    args = parser.parse_args()
    
    if args.find_latest:
        # Find latest results file
        data_dir = Path('./data')
        if not data_dir.exists():
            print("Data directory './data' not found")
            return
        
        pattern = '*_Genetic_evolution.*'
        files = list(data_dir.glob(pattern))
        
        if not files:
            print(f"No results files found matching pattern: {pattern}")
            return
        
        # Sort by modification time and get latest
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        results_file = str(latest_file)
        print(f"Found latest results file: {results_file}")
    else:
        results_file = args.results_file
        
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return
    
    try:
        analyze_optimization_results(
            results_file,
            save_plots=not args.no_save,
            show_plots=not args.no_show
        )
    except Exception as e:
        print(f"Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()