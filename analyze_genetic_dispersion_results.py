#!/usr/bin/env python3
"""
Analysis script for genetic dispersion optimization results.

This script loads and analyzes the results from genetic algorithm optimization
of dispersion profiles for frequency comb generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from datetime import datetime

def load_genetic_results(data_file):
    """Load genetic optimization results from NPZ file."""
    print(f"Loading results from: {data_file}")
    
    try:
        data = np.load(data_file, allow_pickle=True)
        
        # Extract key results
        results = {
            'dispersion_params': data['dispersion_params'],
            'fitness_evolution': data['fitness_evolution'],
            'best_fitness_history': data['best_fitness_history'],
            'diversity_history': data['diversity_history'],
            'final_generation': data['final_generation'],
            'best_individual_idx': data['best_individual_idx'],
            'best_fitness': data['best_fitness'],
            'optimization_params': data['optimization_params'].item(),
            'psi_evolution': data['psi_evolution'] if 'psi_evolution' in data else None,
            'variance_gl': data['variance_gl'] if 'variance_gl' in data else None,
            'convergence_flags': data['convergence_flags'] if 'convergence_flags' in data else None
        }
        
        print(f"✓ Loaded optimization with {results['optimization_params']['n_pop']} individuals")
        print(f"✓ Final generation: {results['final_generation']}")
        print(f"✓ Best fitness: {results['best_fitness']:.6e}")
        
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def analyze_fitness_evolution(results):
    """Analyze fitness evolution over generations."""
    print("\n=== Fitness Evolution Analysis ===")
    
    best_fitness = results['best_fitness_history']
    fitness_evolution = results['fitness_evolution']
    
    # Statistics
    n_gens = len(best_fitness)
    initial_fitness = best_fitness[0] if len(best_fitness) > 0 else np.nan
    final_fitness = best_fitness[-1] if len(best_fitness) > 0 else np.nan
    improvement = (initial_fitness - final_fitness) / initial_fitness if not np.isnan(initial_fitness) and initial_fitness != 0 else 0
    
    print(f"Generations completed: {n_gens}")
    print(f"Initial best fitness: {initial_fitness:.6e}")
    print(f"Final best fitness: {final_fitness:.6e}")
    print(f"Improvement: {improvement*100:.2f}%")
    
    # Convergence analysis
    if len(best_fitness) > 3:
        # Check for stagnation (no improvement over last few generations)
        recent_improvement = np.abs(np.diff(best_fitness[-5:]))
        avg_recent_improvement = np.mean(recent_improvement)
        print(f"Recent improvement rate: {avg_recent_improvement:.6e}")
        
        # Check for convergence trend
        if n_gens > 5:
            slope = np.polyfit(range(len(best_fitness)), np.log10(best_fitness + 1e-15), 1)[0]
            print(f"Log fitness slope (convergence rate): {slope:.6e}")
    
    # Population statistics
    if fitness_evolution.shape[1] > 0:
        final_pop_fitness = fitness_evolution[:, -1]
        print(f"Final population mean fitness: {np.mean(final_pop_fitness):.6e}")
        print(f"Final population std fitness: {np.std(final_pop_fitness):.6e}")
        print(f"Final population fitness range: {np.min(final_pop_fitness):.2e} - {np.max(final_pop_fitness):.2e}")
    
    return {
        'improvement_percent': improvement * 100,
        'convergence_rate': slope if n_gens > 5 else np.nan,
        'final_diversity': np.std(final_pop_fitness) / np.mean(final_pop_fitness) if fitness_evolution.shape[1] > 0 else np.nan
    }

def analyze_population_diversity(results):
    """Analyze population diversity over generations."""
    print("\n=== Population Diversity Analysis ===")
    
    diversity_history = results['diversity_history']
    
    if len(diversity_history) == 0:
        print("❌ No diversity data available")
        return {}
    
    initial_diversity = diversity_history[0]
    final_diversity = diversity_history[-1]
    max_diversity = np.max(diversity_history)
    min_diversity = np.min(diversity_history)
    
    print(f"Initial diversity: {initial_diversity:.4f}")
    print(f"Final diversity: {final_diversity:.4f}")
    print(f"Max diversity: {max_diversity:.4f}")
    print(f"Min diversity: {min_diversity:.4f}")
    
    # Diversity trend
    if len(diversity_history) > 3:
        diversity_slope = np.polyfit(range(len(diversity_history)), diversity_history, 1)[0]
        print(f"Diversity trend slope: {diversity_slope:.6f}")
        
        diversity_status = "increasing" if diversity_slope > 0 else "decreasing"
        print(f"Diversity is {diversity_status} over time")
    
    return {
        'initial_diversity': initial_diversity,
        'final_diversity': final_diversity,
        'diversity_trend': diversity_slope if len(diversity_history) > 3 else 0
    }

def analyze_parameter_evolution(results):
    """Analyze how parameters evolved over generations."""
    print("\n=== Parameter Evolution Analysis ===")
    
    dispersion_params = results['dispersion_params']
    
    if len(dispersion_params) == 0:
        print("❌ No parameter evolution data available")
        return {}
    
    # Extract parameter statistics over generations
    param_stats = {}
    
    for gen_idx, params_data in enumerate(dispersion_params):
        print(f"\nGeneration {gen_idx + 1}:")
        
        # Handle both DataFrame and dict-like structures
        if hasattr(params_data, 'columns'):
            param_names = params_data.columns
        elif isinstance(params_data, dict):
            param_names = params_data.keys()
        else:
            # Skip if we can't parse the parameter structure
            print(f"  ❌ Cannot parse parameter structure for generation {gen_idx + 1}")
            continue
            
        for param_name in param_names:
            if param_name not in param_stats:
                param_stats[param_name] = {
                    'means': [],
                    'stds': [],
                    'mins': [],
                    'maxs': []
                }
            
            # Access parameter values based on data structure
            if hasattr(params_data, 'columns'):
                param_values = params_data[param_name]
            elif isinstance(params_data, dict):
                param_values = params_data[param_name]
            else:
                continue
            
            # Handle different parameter types
            if param_name == 'polyCoefs':
                # For polynomial coefficients, analyze each coefficient
                coeff_means = []
                coeff_stds = []
                for i in range(len(param_values)):
                    if hasattr(param_values.iloc[i], '__len__'):
                        coeff_means.append(np.mean(param_values.iloc[i]))
                        coeff_stds.append(np.std(param_values.iloc[i]))
                
                if coeff_means:
                    param_stats[param_name]['means'].append(np.mean(coeff_means))
                    param_stats[param_name]['stds'].append(np.mean(coeff_stds))
                    print(f"  {param_name}: mean coeff magnitude = {np.mean(coeff_means):.6f}")
            else:
                # Scalar parameters
                try:
                    values = param_values.values
                    param_stats[param_name]['means'].append(np.mean(values))
                    param_stats[param_name]['stds'].append(np.std(values))
                    param_stats[param_name]['mins'].append(np.min(values))
                    param_stats[param_name]['maxs'].append(np.max(values))
                    
                    print(f"  {param_name}: mean = {np.mean(values):.6f}, std = {np.std(values):.6f}")
                except Exception as e:
                    print(f"  {param_name}: analysis failed - {e}")
    
    return param_stats

def analyze_best_individual(results):
    """Analyze the best individual found."""
    print("\n=== Best Individual Analysis ===")
    
    best_idx = results['best_individual_idx']
    best_fitness = results['best_fitness']
    final_gen = results['final_generation'] - 1 if results['final_generation'] > 0 else 0
    
    print(f"Best individual index: {best_idx}")
    print(f"Best fitness: {best_fitness:.6e}")
    
    # Extract best individual parameters from final generation
    if len(results['dispersion_params']) > final_gen:
        best_params_data = results['dispersion_params'][final_gen]
        
        # Handle different parameter data structures
        if hasattr(best_params_data, 'iloc'):
            best_params = best_params_data.iloc[best_idx]
        elif isinstance(best_params_data, dict):
            best_params = {k: v[best_idx] if hasattr(v, '__getitem__') else v for k, v in best_params_data.items()}
        else:
            print("  ❌ Cannot extract best individual parameters")
            best_params = None
        
        if best_params is not None:
            print(f"\nBest individual parameters:")
            for param_name, value in best_params.items():
                if param_name == 'polyCoefs' and hasattr(value, '__len__'):
                    print(f"  {param_name}: [{value[0]:.6f}, {value[1]:.6f}, ..., {value[-1]:.6f}] (length {len(value)})")
                else:
                    print(f"  {param_name}: {value}")
    
    # PSI evolution analysis (if available)
    if results['psi_evolution'] is not None:
        psi_final = results['psi_evolution'][:, :, best_idx, final_gen]
        
        print(f"\nPSI field analysis:")
        print(f"  Shape: {psi_final.shape}")
        print(f"  Max amplitude: {np.max(np.abs(psi_final)):.6f}")
        print(f"  Total power: {np.sum(np.abs(psi_final)**2):.6f}")

def create_analysis_plots(results, output_dir='./analysis_plots'):
    """Create comprehensive analysis plots."""
    print(f"\n=== Creating Analysis Plots ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Fitness Evolution
    plt.figure(figsize=(15, 10))
    
    # Best fitness evolution
    plt.subplot(2, 3, 1)
    best_fitness = results['best_fitness_history']
    plt.semilogy(range(1, len(best_fitness) + 1), best_fitness, 'b-o', markersize=4)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Best Fitness Evolution')
    plt.grid(True, alpha=0.3)
    
    # Population fitness evolution
    plt.subplot(2, 3, 2)
    fitness_evolution = results['fitness_evolution']
    if fitness_evolution.shape[1] > 0:
        generations = range(1, fitness_evolution.shape[1] + 1)
        
        # Plot percentiles
        fitness_25 = np.percentile(fitness_evolution, 25, axis=0)
        fitness_50 = np.percentile(fitness_evolution, 50, axis=0)
        fitness_75 = np.percentile(fitness_evolution, 75, axis=0)
        
        plt.semilogy(generations, fitness_50, 'g-', label='Median', linewidth=2)
        plt.fill_between(generations, fitness_25, fitness_75, alpha=0.3, color='green', label='IQR')
        plt.semilogy(generations, best_fitness, 'b-', label='Best', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness (log scale)')
        plt.title('Population Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Diversity evolution
    plt.subplot(2, 3, 3)
    diversity_history = results['diversity_history']
    if len(diversity_history) > 0:
        plt.plot(range(1, len(diversity_history) + 1), diversity_history, 'r-o', markersize=4)
        plt.xlabel('Generation')
        plt.ylabel('Population Diversity')
        plt.title('Population Diversity Evolution')
        plt.grid(True, alpha=0.3)
    
    # Fitness distribution (final generation)
    plt.subplot(2, 3, 4)
    if fitness_evolution.shape[1] > 0:
        final_fitness = fitness_evolution[:, -1]
        plt.hist(final_fitness, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(results['best_fitness'], color='red', linestyle='--', 
                   label=f'Best: {results["best_fitness"]:.2e}')
        plt.xlabel('Fitness')
        plt.ylabel('Count')
        plt.title('Final Generation Fitness Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Convergence analysis
    plt.subplot(2, 3, 5)
    if len(best_fitness) > 5:
        # Plot convergence rate (improvement per generation)
        improvements = -np.diff(best_fitness)  # Negative because lower fitness is better
        generations = range(2, len(best_fitness) + 1)
        
        plt.semilogy(generations, improvements, 'purple', marker='o', markersize=3)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Improvement (log scale)')
        plt.title('Convergence Rate Analysis')
        plt.grid(True, alpha=0.3)
    
    # Parameter diversity over time
    plt.subplot(2, 3, 6)
    diversity_history = results['diversity_history']
    if len(diversity_history) > 1:
        # Plot overall diversity trend as a proxy for parameter diversity
        plt.plot(range(1, len(diversity_history) + 1), diversity_history, 'orange', marker='s', markersize=4)
        plt.xlabel('Generation')
        plt.ylabel('Population Diversity Metric')
        plt.title('Parameter Diversity Trend')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor parameter analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Parameter Diversity Trend')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'genetic_optimization_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved analysis plots to: {plot_file}")
    
    return plot_file

def main():
    """Main analysis function."""
    print("=== Genetic Dispersion Optimization Analysis ===")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find most recent results file - check multiple possible locations
    possible_dirs = [
        './pyinverse_lle/pyinverse_lle/examples/data',  # Python results first
        '/Users/lucas/Documents/git/github/py-inverseLLE/pyinverse_lle/pyinverse_lle/examples/data',
        './data'  # MATLAB results (fallback)
    ]
    
    data_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir and os.path.exists(data_dir):
        result_files = glob.glob(os.path.join(data_dir, '*Genetic_evolution.npz'))
        if result_files:
            # Use most recent file
            latest_file = max(result_files, key=os.path.getctime)
            print(f"Found {len(result_files)} result files, using latest: {os.path.basename(latest_file)}")
        else:
            print("❌ No genetic evolution results found in data directory")
            return
    else:
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Load and analyze results
    results = load_genetic_results(latest_file)
    if results is None:
        return
    
    # Perform comprehensive analysis
    fitness_analysis = analyze_fitness_evolution(results)
    diversity_analysis = analyze_population_diversity(results)
    param_analysis = analyze_parameter_evolution(results)
    analyze_best_individual(results)
    
    # Create plots
    plot_file = create_analysis_plots(results)
    
    # Summary report
    print(f"\n=== Analysis Summary ===")
    print(f"Optimization Performance:")
    print(f"  - Fitness improvement: {fitness_analysis.get('improvement_percent', 0):.2f}%")
    print(f"  - Final population diversity: {fitness_analysis.get('final_diversity', 0):.4f}")
    print(f"  - Convergence rate: {fitness_analysis.get('convergence_rate', 0):.6e}")
    
    print(f"Population Dynamics:")
    print(f"  - Initial diversity: {diversity_analysis.get('initial_diversity', 0):.4f}")
    print(f"  - Final diversity: {diversity_analysis.get('final_diversity', 0):.4f}")
    print(f"  - Diversity trend: {'increasing' if diversity_analysis.get('diversity_trend', 0) > 0 else 'decreasing'}")
    
    print(f"\nResults saved to: {os.path.dirname(plot_file)}")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()