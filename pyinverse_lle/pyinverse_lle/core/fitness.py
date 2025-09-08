"""
Fitness functions for genetic algorithm optimization.

Translated from MATLAB implementation.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any
from .lle_solver import spectrum_f


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert linear scale to dB."""
    return 10 * np.log10(np.maximum(x, 1e-15))  # Avoid log(0)


def fitness_line_power_target(psi: np.ndarray, disp_param, target_power_db: float, 
                             power_var_tol: float, target_width_modes: int) -> tuple:
    """
    Fitness function targeting specific power levels within a bandwidth.
    
    Parameters
    ----------
    psi : np.ndarray
        Field state(s) to evaluate
    disp_param : DispersionParametrization
        Dispersion parameter object containing mu (mode indices)
    target_power_db : float
        Target power level in dB
    power_var_tol : float  
        Allowed power variation tolerance in dB
    target_width_modes : int
        Target width in number of modes
        
    Returns
    -------
    tuple
        (scores, components_dataframe)
    """
    # Compute power spectrum
    spectr = spectrum_f(psi)
    
    # Region of interest - exclude central mode
    roi = (np.abs(disp_param.mu) <= target_width_modes) & (disp_param.mu != 0)
    
    # Threshold metric - penalize power below target
    def thres_metric(x):
        return np.maximum(0, np.abs(x) - power_var_tol/2)**2
    
    # Power metric - penalty for deviating from target
    power_db = to_db(spectr[roi, :]) if spectr.ndim > 1 else to_db(spectr[roi])
    
    if power_db.ndim == 1:
        power_metric = np.sum(thres_metric(power_db - target_power_db))
        power_metric = np.array([power_metric])
    else:
        power_metric = np.sum(thres_metric(power_db - target_power_db), axis=0)
    
    # Pulse metric - favor soliton-like pulses
    u = np.abs(psi)
    if u.ndim == 1:
        max_u = np.max(u)
        if max_u > 1e-15:  # Avoid division by zero
            pulse_metric = 5 * (np.median(u) + np.median(np.abs(u - np.median(u)))) / max_u
        else:
            pulse_metric = 1e6  # Large penalty for zero field
        pulse_metric = np.array([pulse_metric])
    else:
        pulse_metrics = []
        for i in range(u.shape[1]):
            u_i = u[:, i]
            max_u_i = np.max(u_i)
            if max_u_i > 1e-15:  # Avoid division by zero
                metric = 5 * (np.median(u_i) + np.median(np.abs(u_i - np.median(u_i)))) / max_u_i
            else:
                metric = 1e6  # Large penalty for zero field
            pulse_metrics.append(metric)
        pulse_metric = np.array(pulse_metrics)
    
    # Create components dataframe
    components = pd.DataFrame({
        'powerMetric': power_metric,
        'pulse': pulse_metric
    })
    
    # Total score - currently using only power metric
    scores = power_metric
    
    return scores, components


def fitness_fit_target(psi: np.ndarray, disp_param, comb_target: np.ndarray) -> tuple:
    """
    Fitness function for fitting to a specific target comb shape.
    
    Parameters
    ----------
    psi : np.ndarray
        Field state(s) to evaluate  
    disp_param : DispersionParametrization
        Dispersion parameter object
    comb_target : np.ndarray
        Target comb spectrum
        
    Returns
    -------
    tuple
        (scores, components_dataframe)
    """
    # Compute power spectrum
    spectr = spectrum_f(psi)
    
    # Normalize target
    comb_target_norm = comb_target / np.max(comb_target)
    
    if spectr.ndim == 1:
        spectr_norm = spectr / np.max(spectr)
        # Mean squared error between target and actual
        mse = np.mean((spectr_norm - comb_target_norm)**2)
        scores = np.array([mse])
    else:
        scores = []
        for i in range(spectr.shape[1]):
            spectr_norm = spectr[:, i] / np.max(spectr[:, i])
            mse = np.mean((spectr_norm - comb_target_norm)**2)
            scores.append(mse)
        scores = np.array(scores)
    
    # Create components dataframe
    components = pd.DataFrame({
        'targetFit': scores
    })
    
    return scores, components


def fitness_max_line_power(psi: np.ndarray, disp_param, target_power_db: float,
                          target_width_modes: int) -> tuple:
    """
    Fitness function for maximizing power in specified bandwidth.
    
    Parameters
    ----------
    psi : np.ndarray
        Field state(s) to evaluate
    disp_param : DispersionParametrization
        Dispersion parameter object  
    target_power_db : float
        Target power level in dB
    target_width_modes : int
        Target width in modes
        
    Returns
    -------
    tuple
        (scores, components_dataframe)
    """
    # Compute power spectrum
    spectr = spectrum_f(psi)
    
    # Region of interest
    roi = (np.abs(disp_param.mu) <= target_width_modes) & (disp_param.mu != 0)
    
    if spectr.ndim == 1:
        # Total power in ROI
        total_power = np.sum(spectr[roi])
        # Convert to dB and compare to target
        power_db = to_db(total_power)
        score = np.abs(power_db - target_power_db)
        scores = np.array([score])
    else:
        scores = []
        for i in range(spectr.shape[1]):
            total_power = np.sum(spectr[roi, i])
            power_db = to_db(total_power)
            score = np.abs(power_db - target_power_db)
            scores.append(score)
        scores = np.array(scores)
    
    components = pd.DataFrame({
        'powerTarget': scores
    })
    
    return scores, components


def fitness_super_octave(psi: np.ndarray, disp_param, octave_modes: Union[int, list]) -> tuple:
    """
    Fitness function for octave-spanning comb optimization.
    
    Parameters
    ----------
    psi : np.ndarray
        Field state(s) to evaluate
    disp_param : DispersionParametrization
        Dispersion parameter object
    octave_modes : int or list
        Target octave mode(s)
        
    Returns
    -------
    tuple
        (scores, components_dataframe)
    """
    # Compute power spectrum
    spectr = spectrum_f(psi)
    
    if not isinstance(octave_modes, (list, np.ndarray)):
        octave_modes = [octave_modes]
    
    if spectr.ndim == 1:
        # Sum power at octave modes
        octave_power = 0
        for mode in octave_modes:
            mode_idx = np.argmin(np.abs(disp_param.mu - mode))
            octave_power += spectr[mode_idx]
        
        # Penalty: negative of octave power (to maximize)
        scores = np.array([-to_db(octave_power)])
    else:
        scores = []
        for i in range(spectr.shape[1]):
            octave_power = 0
            for mode in octave_modes:
                mode_idx = np.argmin(np.abs(disp_param.mu - mode))
                octave_power += spectr[mode_idx, i]
            scores.append(-to_db(octave_power))
        scores = np.array(scores)
    
    components = pd.DataFrame({
        'octavePower': -scores  # Store positive values
    })
    
    return scores, components


def adaptive_fitness_weights(generation: int, max_generations: int) -> Dict[str, float]:
    """
    Compute adaptive weights for multi-objective fitness functions.
    
    Parameters
    ----------
    generation : int
        Current generation number
    max_generations : int
        Maximum number of generations
        
    Returns
    -------
    dict
        Fitness component weights
    """
    progress = generation / max_generations
    
    # Early generations: focus on basic power requirements
    # Later generations: fine-tune spectrum shape
    power_weight = 1.0
    shape_weight = progress * 0.5
    stability_weight = progress * 0.3
    
    return {
        'power': power_weight,
        'shape': shape_weight,
        'stability': stability_weight
    }


class FitnessEvaluator:
    """
    Unified fitness evaluator supporting multiple objectives.
    """
    
    def __init__(self, fitness_type: str, **kwargs):
        """
        Initialize fitness evaluator.
        
        Parameters
        ----------
        fitness_type : str
            Type of fitness function ('line_power_target', 'fit_target', etc.)
        **kwargs
            Additional parameters for fitness function
        """
        self.fitness_type = fitness_type
        self.params = kwargs
        
        # Map fitness functions
        self.fitness_functions = {
            'line_power_target': fitness_line_power_target,
            'fit_target': fitness_fit_target,
            'max_line_power': fitness_max_line_power,
            'super_octave': fitness_super_octave
        }
    
    def evaluate(self, psi: np.ndarray, disp_param) -> tuple:
        """
        Evaluate fitness for given field state(s).
        
        Parameters
        ----------
        psi : np.ndarray
            Field state(s) to evaluate
        disp_param : DispersionParametrization
            Dispersion parameter object
            
        Returns
        -------
        tuple
            (scores, components)
        """
        if self.fitness_type not in self.fitness_functions:
            raise ValueError(f"Unknown fitness type: {self.fitness_type}")
        
        fitness_func = self.fitness_functions[self.fitness_type]
        
        # Call appropriate fitness function with parameters
        return fitness_func(psi, disp_param, **self.params)
    
    def evaluate_population(self, psi_population: np.ndarray, disp_param) -> np.ndarray:
        """
        Evaluate fitness for entire population.
        
        Parameters
        ----------
        psi_population : np.ndarray
            Population of field states (n_modes, n_directions, n_pop)
        disp_param : DispersionParametrization
            Dispersion parameter object
            
        Returns
        -------
        np.ndarray
            Fitness scores for population
        """
        n_pop = psi_population.shape[-1]
        n_dir = psi_population.shape[1] if psi_population.ndim == 3 else 1
        
        all_scores = []
        
        for i_pop in range(n_pop):
            if psi_population.ndim == 3:
                # Multiple directions
                dir_scores = []
                for i_dir in range(n_dir):
                    psi_single = psi_population[:, i_dir, i_pop]
                    scores, _ = self.evaluate(psi_single, disp_param)
                    dir_scores.append(scores[0] if hasattr(scores, '__len__') else scores)
                # Take minimum (best) score across directions
                final_score = min(dir_scores)
            else:
                # Single direction
                psi_single = psi_population[:, i_pop]
                scores, _ = self.evaluate(psi_single, disp_param)
                final_score = scores[0] if hasattr(scores, '__len__') else scores
            
            all_scores.append(final_score)
        
        return np.array(all_scores)