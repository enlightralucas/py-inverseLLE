"""
Visualization utilities for LLE optimization results.

Provides plotting functions equivalent to MATLAB plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from typing import Optional, Union, List, Tuple
from ..core.lle_solver import spectrum_f


def plot_dispersion(mu: np.ndarray, dispersion: np.ndarray, 
                   title: str = 'Dispersion Profile', 
                   xlabel: str = 'Mode #', ylabel: str = 'Dispersion',
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot dispersion profile(s).
    
    Parameters
    ----------
    mu : np.ndarray
        Mode indices
    dispersion : np.ndarray
        Dispersion values (1D or 2D for multiple profiles)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label  
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if dispersion.ndim == 1:
        ax.plot(mu, dispersion, 'b-', linewidth=2)
    else:
        # Multiple profiles
        n_profiles = dispersion.shape[1] if dispersion.shape[1] < dispersion.shape[0] else dispersion.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
        
        for i in range(n_profiles):
            profile = dispersion[:, i] if dispersion.shape[1] == n_profiles else dispersion[i, :]
            ax.plot(mu, profile, color=colors[i], alpha=0.7, linewidth=1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    return ax


def plot_spectrum(mu: np.ndarray, spectrum: np.ndarray,
                 title: str = 'Spectrum', 
                 xlabel: str = 'Mode #', ylabel: str = 'Power',
                 log_scale: bool = False, db_scale: bool = False,
                 ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot power spectrum.
    
    Parameters
    ----------
    mu : np.ndarray
        Mode indices
    spectrum : np.ndarray
        Spectrum values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    log_scale : bool
        Use logarithmic y-scale
    db_scale : bool
        Convert to dB scale
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = spectrum.copy()
    
    if db_scale:
        plot_data = 10 * np.log10(np.maximum(plot_data, 1e-15))
        ylabel += ' (dB)'
    
    ax.plot(mu, plot_data, 'b-', linewidth=2)
    
    if log_scale and not db_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_stem_spectrum(mu: np.ndarray, spectrum: np.ndarray,
                      title: str = 'Comb Spectrum',
                      xlabel: str = 'Mode #', ylabel: str = 'Power (dB)',
                      xlim: Optional[Tuple[float, float]] = None,
                      ylim: Optional[Tuple[float, float]] = None,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot comb spectrum as stem plot.
    
    Parameters
    ----------
    mu : np.ndarray
        Mode indices
    spectrum : np.ndarray
        Spectrum values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
        Plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to dB
    spectrum_db = 10 * np.log10(np.maximum(spectrum, 1e-15))
    spectrum_db = np.maximum(spectrum_db, -100)  # Floor at -100 dB
    
    ax.stem(mu, spectrum_db, basefmt=' ', linefmt='b-', markerfmt='none')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    return ax


def plot_evolution_summary(results: dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive evolution summary plot.
    
    Parameters
    ----------
    results : dict
        Optimization results dictionary
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Figure handle
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Fitness evolution
    plt.subplot(3, 3, 1)
    plt.semilogy(results['best_fitness_history'], 'b-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Evolution')
    plt.grid(True, alpha=0.3)
    
    # Mean fitness evolution
    plt.subplot(3, 3, 2)
    mean_fitness = np.mean(results['fitness_evolution'], axis=0)
    std_fitness = np.std(results['fitness_evolution'], axis=0)
    
    generations = np.arange(len(mean_fitness))
    plt.semilogy(generations, mean_fitness, 'r-', linewidth=2, label='Mean')
    plt.fill_between(generations, 
                    mean_fitness - std_fitness,
                    mean_fitness + std_fitness,
                    alpha=0.3, color='red', label='±1 std')
    plt.semilogy(results['best_fitness_history'], 'b-', linewidth=2, label='Best')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Population Fitness Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Diversity evolution
    if len(results['diversity_history']) > 0:
        plt.subplot(3, 3, 3)
        plt.plot(results['diversity_history'], 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Population Diversity')
        plt.title('Population Diversity Evolution')
        plt.grid(True, alpha=0.3)
    
    # Convergence flags
    plt.subplot(3, 3, 4)
    convergence_rate = np.mean(results['convergence_flags'], axis=0)
    plt.plot(convergence_rate, 'purple', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Convergence Rate')
    plt.title('Population Convergence Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    
    # Stability metric
    plt.subplot(3, 3, 5)
    mean_stability = np.mean(results['variance_gl'], axis=0)
    std_stability = np.std(results['variance_gl'], axis=0)
    
    plt.semilogy(mean_stability, 'orange', linewidth=2, label='Mean')
    plt.fill_between(np.arange(len(mean_stability)),
                    mean_stability - std_stability,
                    mean_stability + std_stability,
                    alpha=0.3, color='orange')
    
    plt.xlabel('Generation')
    plt.ylabel('Stability Metric')
    plt.title('Solution Stability Evolution')
    plt.grid(True, alpha=0.3)
    
    # Final fitness distribution
    plt.subplot(3, 3, 6)
    final_fitness = results['fitness_evolution'][:, -1]
    plt.hist(final_fitness, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(results['best_fitness'], color='red', linestyle='--', linewidth=2,
               label=f'Best: {results["best_fitness"]:.2e}')
    plt.xlabel('Fitness')
    plt.ylabel('Count')
    plt.title('Final Generation Fitness Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parameter evolution (if polynomial coefficients available)
    if 'dispersion_params' in results and len(results['dispersion_params']) > 0:
        plt.subplot(3, 3, 7)
        
        # Plot evolution of first few polynomial coefficients
        n_gens = len(results['dispersion_params'])
        n_coeffs_to_plot = min(3, results['dispersion_params'][0].shape[1] 
                              if 'polyCoefs' in results['dispersion_params'][0].columns else 0)
        
        if n_coeffs_to_plot > 0:
            colors = ['red', 'blue', 'green']
            for i in range(n_coeffs_to_plot):
                coeff_evolution = []
                for gen in range(n_gens):
                    if 'polyCoefs' in results['dispersion_params'][gen].columns:
                        best_idx = np.argmin(results['fitness_evolution'][:, gen])
                        poly_coeffs = results['dispersion_params'][gen].loc[best_idx, 'polyCoefs']
                        if hasattr(poly_coeffs, '__len__') and len(poly_coeffs) > i:
                            coeff_evolution.append(poly_coeffs[i])
                        else:
                            coeff_evolution.append(0)
                    else:
                        coeff_evolution.append(0)
                
                plt.plot(coeff_evolution, color=colors[i], linewidth=2, 
                        label=f'Coeff {i}')
            
            plt.xlabel('Generation')
            plt.ylabel('Coefficient Value')
            plt.title('Best Individual Parameter Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Success rate over generations
    plt.subplot(3, 3, 8)
    fitness_threshold = results['optimization_params']['fitness_tol'] * 10  # 10x tolerance
    success_rate = np.mean(results['fitness_evolution'] < fitness_threshold, axis=0)
    plt.plot(success_rate * 100, 'brown', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rate (Fitness < {fitness_threshold:.1e})')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    
    # Optimization parameters summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    params_text = f"""Optimization Parameters:
Population Size: {results['optimization_params']['n_pop']}
Max Generations: {results['optimization_params']['max_iter']}
Completed Generations: {results['final_generation']}
Fitness Tolerance: {results['optimization_params']['fitness_tol']:.1e}
Best Fitness: {results['best_fitness']:.3e}
Mutation Probability: {results['optimization_params']['mutation_proba']}
Number of Modes: {results['optimization_params']['n_modes']}

Status: {'CONVERGED' if results['best_fitness'] < results['optimization_params']['fitness_tol'] else 'NOT CONVERGED'}
"""
    
    plt.text(0.1, 0.9, params_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution summary saved to: {save_path}")
    
    return fig


def plot_comparison_with_target(mu: np.ndarray, target_spectrum: np.ndarray,
                               optimized_spectrum: np.ndarray,
                               title: str = 'Target vs Optimized Spectrum',
                               xlim: Optional[Tuple[float, float]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison between target and optimized spectra.
    
    Parameters
    ----------
    mu : np.ndarray
        Mode indices
    target_spectrum : np.ndarray
        Target spectrum
    optimized_spectrum : np.ndarray
        Optimized spectrum
    title : str
        Plot title
    xlim : tuple, optional
        X-axis limits
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Figure handle
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Linear scale comparison
    ax1.plot(mu, target_spectrum, 'r--', linewidth=3, label='Target', alpha=0.8)
    ax1.plot(mu, optimized_spectrum, 'b-', linewidth=2, label='Optimized')
    ax1.set_xlabel('Mode #')
    ax1.set_ylabel('Power (linear)')
    ax1.set_title(f'{title} - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if xlim:
        ax1.set_xlim(xlim)
    
    # dB scale comparison
    target_db = 10 * np.log10(np.maximum(target_spectrum, 1e-15))
    optimized_db = 10 * np.log10(np.maximum(optimized_spectrum, 1e-15))
    
    ax2.plot(mu, target_db, 'r--', linewidth=3, label='Target', alpha=0.8)
    ax2.plot(mu, optimized_db, 'b-', linewidth=2, label='Optimized')
    ax2.set_xlabel('Mode #')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title(f'{title} - dB Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-80, 10])
    if xlim:
        ax2.set_xlim(xlim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_population_dispersion(mu: np.ndarray, dispersion_population: np.ndarray,
                              title: str = 'Population Dispersion Profiles',
                              n_plot: int = 50) -> plt.Figure:
    """
    Plot dispersion profiles for population sample.
    
    Parameters
    ----------
    mu : np.ndarray
        Mode indices
    dispersion_population : np.ndarray
        Dispersion profiles for population (n_modes x n_pop)
    title : str
        Plot title
    n_plot : int
        Number of profiles to plot
        
    Returns
    -------
    plt.Figure
        Figure handle
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_pop = dispersion_population.shape[1]
    n_to_plot = min(n_plot, n_pop)
    
    # Create HSV colormap for population
    colors = plt.cm.hsv(np.linspace(0, 1, n_to_plot))
    
    # Plot sample of population
    indices = np.linspace(0, n_pop-1, n_to_plot, dtype=int)
    
    for i, idx in enumerate(indices):
        ax.plot(mu, dispersion_population[:, idx], color=colors[i], 
               alpha=0.7, linewidth=1)
    
    # Highlight first (seed) individual
    ax.plot(mu, dispersion_population[:, 0], 'k-', linewidth=3, 
           alpha=0.8, label='Seed Individual')
    
    ax.set_xlabel('Mode #')
    ax.set_ylabel('Dispersion')
    ax.set_title(f'{title} (showing {n_to_plot}/{n_pop} individuals)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    return fig


def animate_evolution(results: dict, mu: np.ndarray, 
                     save_path: Optional[str] = None) -> None:
    """
    Create animation of evolution progress (requires matplotlib animation).
    
    Parameters
    ----------
    results : dict
        Optimization results
    mu : np.ndarray
        Mode indices
    save_path : str, optional
        Path to save animation
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Animation requires matplotlib.animation - skipping")
        return
    
    # This would create an animated visualization of the evolution
    # Implementation would depend on specific requirements
    print("Animation feature not fully implemented yet")


# Utility functions for specific plot types
def vline(x: float, **kwargs) -> None:
    """Add vertical line to current plot (MATLAB vline equivalent)."""
    plt.axvline(x, **kwargs)


def hline(y: float, **kwargs) -> None:
    """Add horizontal line to current plot (MATLAB hline equivalent).""" 
    plt.axhline(y, **kwargs)


def shade_area(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, 
              alpha: float = 0.3, **kwargs) -> None:
    """Shade area between curves."""
    plt.fill_between(x, y1, y2, alpha=alpha, **kwargs)