"""
Population management for genetic algorithm.

Translated from MATLAB create_new_pop.m
"""

import numpy as np
from typing import Dict, Any


def create_new_population(param_arrays: Dict[str, np.ndarray], fitness_array: np.ndarray, 
                         mutation_proba: float) -> Dict[str, np.ndarray]:
    """
    Create new population based on fitness-based selection and mutation.
    
    Parameters
    ----------
    param_arrays : Dict[str, np.ndarray]
        Dictionary of parameter arrays for each parameter type
    fitness_array : np.ndarray
        Fitness values for current population (lower is better)
    mutation_proba : float
        Mutation probability
        
    Returns
    -------
    Dict[str, np.ndarray]
        New population parameter arrays
    """
    n_pop = len(fitness_array)
    
    # Compute average parameter based on fitness weighting
    average_params = {}
    for param_name, param_array in param_arrays.items():
        try:
            if param_array.ndim == 1 and not (hasattr(param_array, 'dtype') and param_array.dtype == object):
                # Simple 1D parameter
                weights = fitness_array / np.sum(fitness_array)
                average_params[param_name] = np.sum(weights * param_array)
            elif hasattr(param_array, 'dtype') and param_array.dtype == object:
                # Handle object arrays - compute average for each component
                try:
                    # Try to create a representative average
                    first_elem = param_array[0]
                    if hasattr(first_elem, '__len__'):
                        avg_length = len(first_elem)
                        avg_array = np.zeros(avg_length)
                        total_weight = 0.0
                        for i in range(len(param_array)):
                            if hasattr(param_array[i], '__len__') and len(param_array[i]) == avg_length:
                                avg_array += param_array[i] * fitness_array[i]
                                total_weight += fitness_array[i]
                        if total_weight > 0:
                            average_params[param_name] = avg_array / total_weight
                        else:
                            average_params[param_name] = first_elem.copy()
                    else:
                        # Object array with scalar elements
                        weights = fitness_array / np.sum(fitness_array)
                        avg_val = sum(param_array[i] * weights[i] for i in range(len(param_array)))
                        average_params[param_name] = avg_val
                except Exception:
                    # Fallback to first element
                    average_params[param_name] = param_array[0] if hasattr(param_array[0], 'copy') else param_array[0]
            elif hasattr(param_array, 'ndim') and param_array.ndim == 2:
                # 2D arrays
                weights = fitness_array.reshape(-1, 1) / np.sum(fitness_array)
                if weights.shape[0] == param_array.shape[0]:
                    average_params[param_name] = np.sum(weights * param_array, axis=0)
                else:
                    # Shape mismatch - use simple mean
                    average_params[param_name] = np.mean(param_array, axis=0)
            else:
                # Handle more complex parameter structures
                try:
                    average_params[param_name] = np.mean(param_array, axis=0)
                except Exception:
                    # Fallback for irregular structures
                    average_params[param_name] = param_array[0]
        except Exception:
            # Ultimate fallback
            average_params[param_name] = param_array[0] if len(param_array) > 0 else 0.0
    
    # Rank individuals by fitness (ascending order - lower fitness is better)
    order = np.argsort(fitness_array)
    ranking = np.zeros(len(fitness_array))
    ranking[order] = np.arange(1, len(fitness_array) + 1)
    
    # Convert ranking to selection probability (higher rank = lower fitness = higher selection prob)
    score = 1.0 / np.sqrt(ranking)
    score = 2 * n_pop / np.sum(score) * score
    
    def make_cumulative_distribution(scores):
        """Create cumulative distribution for selection."""
        normalized = scores / np.sum(scores)
        return np.concatenate([[0], np.cumsum(normalized)])
    
    cum_dist = make_cumulative_distribution(score)
    
    # Initialize new population arrays
    new_param_arrays = {}
    for param_name, param_array in param_arrays.items():
        new_param_arrays[param_name] = np.zeros_like(param_array)
    
    # Keep the fittest individual unchanged
    n_hand_picked = 1
    best_idx = np.argmax(score)
    for param_name in param_arrays.keys():
        if param_arrays[param_name].ndim == 1:
            new_param_arrays[param_name][0] = param_arrays[param_name][best_idx]
        else:
            new_param_arrays[param_name][0] = param_arrays[param_name][best_idx].copy()
    
    # Generate rest of population through crossover and mutation
    n_parents = 3
    
    for i in range(int(np.ceil(n_pop / n_parents))):
        # Select parents using fitness proportionate selection
        parent_probs = np.random.rand(n_parents)
        parent_indices = np.digitize(parent_probs, cum_dist) - 1
        parent_indices = np.clip(parent_indices, 0, n_pop - 1)
        
        # Extract parent parameters
        selected_parents = {}
        for param_name, param_array in param_arrays.items():
            if param_array.ndim == 1:
                selected_parents[param_name] = param_array[parent_indices]
            else:
                selected_parents[param_name] = param_array[parent_indices]
        
        # Create offspring for each parent slot
        for j in range(n_parents):
            k = i * n_parents + j + n_hand_picked
            if k >= n_pop:
                break
            
            # Crossover: select parent for each gene/parameter component
            for param_name, parent_params in selected_parents.items():
                if param_arrays[param_name].ndim == 1:
                    # Simple parameter - just select one parent
                    parent_weights = score[parent_indices]
                    parent_idx = np.random.choice(n_parents, p=parent_weights/np.sum(parent_weights))
                    offspring_param = parent_params[parent_idx]
                else:
                    # Complex parameter - gene-wise crossover
                    param_length = parent_params.shape[1] if parent_params.ndim > 1 else len(parent_params[0])
                    offspring_param = np.zeros(param_length)
                    
                    # For each gene, select parent based on weighted probability
                    parent_weights = score[parent_indices]
                    for gene_idx in range(param_length):
                        parent_choice = np.random.choice(n_parents, p=parent_weights/np.sum(parent_weights))
                        if parent_params.ndim > 1:
                            offspring_param[gene_idx] = parent_params[parent_choice, gene_idx]
                        else:
                            offspring_param[gene_idx] = parent_params[parent_choice][gene_idx]
                
                # Apply mutation
                if np.random.rand() >= 0.5:
                    # Proportional mutation
                    if param_arrays[param_name].ndim == 1:
                        if np.random.rand() <= mutation_proba:
                            mutation_factor = 1 + mutation_proba * np.random.randn()
                            offspring_param *= mutation_factor
                    else:
                        mutation_mask = np.random.rand(len(offspring_param)) <= mutation_proba
                        mutation_factors = 1 + mutation_proba * np.random.randn(len(offspring_param))
                        offspring_param[mutation_mask] *= mutation_factors[mutation_mask]
                else:
                    # Replacement mutation with average-based values
                    if param_arrays[param_name].ndim == 1:
                        if np.random.rand() <= mutation_proba:
                            offspring_param = average_params[param_name] * np.random.randn()
                    else:
                        mutation_mask = np.random.rand(len(offspring_param)) <= mutation_proba
                        avg_param = average_params[param_name]
                        if hasattr(avg_param, '__len__'):
                            replacement_values = avg_param * np.random.randn(len(offspring_param))
                            offspring_param[mutation_mask] = replacement_values[mutation_mask]
                
                # Store offspring in new population
                if param_arrays[param_name].ndim == 1:
                    new_param_arrays[param_name][k] = offspring_param
                else:
                    new_param_arrays[param_name][k] = offspring_param
    
    return new_param_arrays


class PopulationManager:
    """
    Population manager for genetic algorithm operations.
    """
    
    def __init__(self, population_size: int):
        """
        Initialize population manager.
        
        Parameters
        ---------- 
        population_size : int
            Size of population
        """
        self.population_size = population_size
        self.generation = 0
        
    def tournament_selection(self, fitness_array: np.ndarray, tournament_size: int = 3) -> int:
        """
        Tournament selection for parent choice.
        
        Parameters
        ----------
        fitness_array : np.ndarray
            Fitness values (lower is better)
        tournament_size : int
            Number of individuals in tournament
            
        Returns
        -------
        int
            Index of selected individual
        """
        tournament_indices = np.random.choice(len(fitness_array), tournament_size, replace=False)
        tournament_fitness = fitness_array[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return winner_idx
    
    def roulette_wheel_selection(self, fitness_array: np.ndarray) -> int:
        """
        Roulette wheel selection (fitness proportionate).
        
        Parameters
        ----------
        fitness_array : np.ndarray
            Fitness values (lower is better)
            
        Returns
        -------
        int
            Index of selected individual
        """
        # Convert to selection probabilities (inverse fitness)
        min_fitness = np.min(fitness_array)
        if min_fitness < 0:
            shifted_fitness = fitness_array - min_fitness + 1e-10
        else:
            shifted_fitness = 1.0 / (fitness_array + 1e-10)
        
        probabilities = shifted_fitness / np.sum(shifted_fitness)
        return np.random.choice(len(fitness_array), p=probabilities)
    
    def adaptive_mutation_rate(self, base_rate: float, diversity_metric: float) -> float:
        """
        Compute adaptive mutation rate based on population diversity.
        
        Parameters
        ----------
        base_rate : float
            Base mutation rate
        diversity_metric : float
            Population diversity measure (0-1)
            
        Returns
        -------
        float
            Adaptive mutation rate
        """
        # Increase mutation rate when diversity is low
        diversity_factor = 1.0 + (1.0 - diversity_metric) * 2.0
        return min(base_rate * diversity_factor, 0.5)  # Cap at 50%
    
    def compute_diversity(self, param_arrays: Dict[str, np.ndarray]) -> float:
        """
        Compute population diversity metric.
        
        Parameters
        ----------
        param_arrays : Dict[str, np.ndarray]
            Parameter arrays for population
            
        Returns
        -------
        float
            Diversity metric (0-1, higher = more diverse)
        """
        diversity_measures = []
        
        for param_name, param_array in param_arrays.items():
            try:
                if param_array.ndim == 1 and not (hasattr(param_array, 'dtype') and param_array.dtype == object):
                    # Simple parameter
                    std_dev = np.std(param_array)
                    mean_val = np.mean(np.abs(param_array))
                    if np.isscalar(mean_val) and mean_val > 0:
                        diversity_measures.append(std_dev / mean_val)
                elif hasattr(param_array, 'dtype') and param_array.dtype == object:
                    # Handle object arrays (like polyCoefs with numpy arrays in each cell)
                    try:
                        # Try to extract components from object array
                        first_elem = param_array[0]
                        if hasattr(first_elem, '__len__'):
                            for i in range(len(first_elem)):
                                component = np.array([param_array[j][i] for j in range(len(param_array)) if hasattr(param_array[j], '__len__') and len(param_array[j]) > i])
                                if len(component) > 0:
                                    std_dev = np.std(component)
                                    mean_val = np.mean(np.abs(component))
                                    # Ensure mean_val is scalar before comparison
                                    if np.isscalar(mean_val) and mean_val > 0:
                                        diversity_measures.append(std_dev / mean_val)
                        else:
                            # Object array with scalar elements
                            component = np.array([param_array[j] for j in range(len(param_array))])
                            std_dev = np.std(component)
                            mean_val = np.mean(np.abs(component))
                            if np.isscalar(mean_val) and mean_val > 0:
                                diversity_measures.append(std_dev / mean_val)
                    except Exception:
                        # Skip this parameter if extraction fails
                        continue
                else:
                    # Complex parameter - compute diversity for each component
                    for i in range(param_array.shape[1] if param_array.ndim > 1 else len(param_array[0])):
                        if param_array.ndim > 1:
                            component = param_array[:, i]
                        else:
                            try:
                                component = np.array([param_array[j][i] for j in range(len(param_array))])
                            except (IndexError, TypeError):
                                continue
                        
                        std_dev = np.std(component)
                        mean_val = np.mean(np.abs(component))
                        # Ensure mean_val is scalar before comparison
                        if np.isscalar(mean_val) and mean_val > 0:
                            diversity_measures.append(std_dev / mean_val)
            except Exception:
                # Skip parameter if any error occurs
                continue
        
        # Return average diversity across all parameters
        return np.mean(diversity_measures) if diversity_measures else 0.0