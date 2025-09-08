"""
Main genetic algorithm implementation for LLE dispersion optimization.

Translated from MATLAB lle_dispersion_genetic_optimize.m
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
from typing import Optional, Callable, Dict, Any, Tuple
from joblib import Parallel, delayed
import warnings

from ..core.dispersion import DispersionParametrizationPoly
from ..core.lle_solver import LLEPropagator, AdaptiveTimeStep
from ..core.fitness import FitnessEvaluator
from .population import create_new_population, PopulationManager


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for LLE dispersion engineering.
    """
    
    def __init__(self, dispersion_param: DispersionParametrizationPoly, 
                 fitness_evaluator: FitnessEvaluator, **kwargs):
        """
        Initialize genetic optimizer.
        
        Parameters
        ----------
        dispersion_param : DispersionParametrizationPoly
            Dispersion parametrization object
        fitness_evaluator : FitnessEvaluator
            Fitness evaluation object
        **kwargs
            Additional optimization parameters
        """
        self.disp_param = dispersion_param
        self.fitness_evaluator = fitness_evaluator
        
        # GA parameters
        self.n_pop = kwargs.get('n_pop', 208)
        self.max_iter = kwargs.get('max_iter', 300)
        self.fitness_tol = kwargs.get('fitness_tol', 1e-6)
        self.mutation_proba = kwargs.get('mutation_proba', 0.15)
        
        # LLE parameters
        self.n_modes = len(dispersion_param.mu)
        self.lle_type = kwargs.get('lle_type', 1)
        self.noise_factor = kwargs.get('noise_factor', 0.0)
        
        # Parallel processing
        self.n_workers = kwargs.get('n_workers', -1)  # Use all cores by default
        self.testing_mode = kwargs.get('testing_mode', False)
        
        # Initialize components
        self.lle_propagator = LLEPropagator(self.n_modes)
        self.population_manager = PopulationManager(self.n_pop)
        
        # Storage for results
        self.reset_storage()
        
        # Initialize pulse profile function (to be set by user)
        self.pulse_profile_init = kwargs.get('pulse_profile_init', None)
        
    def reset_storage(self):
        """Reset storage arrays for new optimization run."""
        self.disp_param_storage = []
        self.psi_storage = np.zeros((self.n_modes, self.lle_type, self.n_pop, self.max_iter), 
                                   dtype=complex)
        self.var_gl_storage = np.zeros((self.n_pop, self.max_iter))
        self.conv_flag_storage = np.zeros((self.n_pop, self.max_iter), dtype=bool)
        self.fitness_storage = np.full((self.n_pop, self.max_iter), self.fitness_tol)
        
        self.current_generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        
    def _initialize_pulse(self, disp_param_idx: int, f0: float, 
                         start_branch: str = 'bottom', excite: str = 'soliton') -> np.ndarray:
        """
        Initialize pulse for LLE propagation.
        
        Parameters
        ----------
        disp_param_idx : int
            Index in dispersion parameter population
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
        if self.pulse_profile_init is not None:
            return self.pulse_profile_init.copy()
        
        # Simple initialization - could be made more sophisticated
        if excite == 'soliton':
            # Soliton-like initial condition
            t_range = np.linspace(-10, 10, self.n_modes)
            psi0 = f0 * np.sqrt(2) * np.sech(t_range) * np.exp(1j * 0.5 * t_range**2)
        else:
            # Noise-seeded initialization
            psi0 = np.random.randn(self.n_modes) + 1j * np.random.randn(self.n_modes)
            psi0 *= f0 / np.max(np.abs(psi0))
        
        return psi0
    
    def _propagate_single_individual(self, args: tuple) -> tuple:
        """
        Propagate LLE for a single population member.
        
        Parameters
        ---------- 
        args : tuple
            (population_idx, generation) tuple
            
        Returns
        -------
        tuple
            (psi_final, var_gl, conv_flag, population_idx)
        """
        i_pop, i_gen = args
        
        try:
            # Get dispersion for this individual
            dint = self.disp_param.compute_dispersion(i_pop)
            dispersion = -1.0 - 1j * np.fft.ifftshift(dint)  # Normalized units
            
            # Get parameters for this individual
            zeta = self.disp_param.param_table.loc[i_pop, 'detuning']
            f0 = self.disp_param.param_table.loc[i_pop, 'pumpPow']
            
            # Initialize pulse
            psi0 = self._initialize_pulse(i_pop, f0)
            
            # First propagation - coarse evolution
            t_evol1 = 40.0
            n_store1 = 1000
            h1 = 2**(-9)
            
            psi_tmp, _, var_gl = self.lle_propagator.propagate(
                psi0, f0, zeta, dispersion, self.noise_factor,
                t_evol1, h1, n_store1
            )
            
            # Refinement - adaptive time step
            t_evol2 = 10.0
            n_store2 = 1000
            adaptive_step_obj = AdaptiveTimeStep(h1, 2**(-14), 15, 0.4)
            adaptive_step = lambda t: adaptive_step_obj(t, t_evol2)
            
            psi_final, _, _ = self.lle_propagator.propagate(
                psi_tmp, f0, zeta, dispersion, self.noise_factor,
                t_evol2, adaptive_step, n_store2
            )
            
            # Check convergence (simplified criterion)
            conv_flag = var_gl < 0.1
            
            return psi_final, var_gl, conv_flag, i_pop
            
        except Exception as e:
            # Return failed state
            warnings.warn(f"LLE propagation failed for individual {i_pop}: {str(e)}")
            psi_failed = np.zeros(self.n_modes, dtype=complex)
            return psi_failed, 1e6, False, i_pop
    
    def _evaluate_generation(self, i_gen: int) -> np.ndarray:
        """
        Evaluate fitness for entire generation.
        
        Parameters
        ----------
        i_gen : int
            Generation index
            
        Returns
        -------
        np.ndarray
            Fitness scores for population
        """
        if self.testing_mode:
            # Sequential execution for testing
            pop_range = [0]  # Test only first individual
        else:
            pop_range = list(range(self.n_pop))
        
        # Prepare arguments for parallel execution
        args_list = [(i_pop, i_gen) for i_pop in pop_range]
        
        if self.testing_mode or self.n_workers == 1:
            # Sequential execution
            results = [self._propagate_single_individual(args) for args in args_list]
        else:
            # Parallel execution
            results = Parallel(n_jobs=self.n_workers)(
                delayed(self._propagate_single_individual)(args) for args in args_list
            )
        
        # Process results
        fitness_scores = np.full(self.n_pop, 1e6)  # Default high fitness (bad)
        
        for psi_final, var_gl, conv_flag, i_pop in results:
            # Store results
            if psi_final.ndim == 1:
                self.psi_storage[:, 0, i_pop, i_gen] = psi_final
            else:
                self.psi_storage[:, :, i_pop, i_gen] = psi_final
                
            self.var_gl_storage[i_pop, i_gen] = var_gl
            self.conv_flag_storage[i_pop, i_gen] = conv_flag
            
            # Evaluate fitness
            try:
                if psi_final.ndim == 1:
                    scores, _ = self.fitness_evaluator.evaluate(psi_final, self.disp_param)
                    fitness_scores[i_pop] = scores[0] if hasattr(scores, '__len__') else scores
                else:
                    # Multiple directions - evaluate each and take minimum
                    dir_scores = []
                    for i_dir in range(psi_final.shape[1]):
                        scores, _ = self.fitness_evaluator.evaluate(psi_final[:, i_dir], self.disp_param)
                        dir_scores.append(scores[0] if hasattr(scores, '__len__') else scores)
                    fitness_scores[i_pop] = min(dir_scores)
            except Exception as e:
                warnings.warn(f"Fitness evaluation failed for individual {i_pop}: {str(e)}")
                fitness_scores[i_pop] = 1e6
        
        return fitness_scores
    
    def optimize(self, save_data: bool = True, output_dir: str = './data') -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Parameters
        ----------
        save_data : bool
            Whether to save optimization data
        output_dir : str
            Directory for output files
            
        Returns
        -------
        dict
            Optimization results
        """
        print("Starting genetic algorithm optimization...")
        print(f"Population size: {self.n_pop}, Max generations: {self.max_iter}")
        
        # Create output directory
        if save_data:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y_%m_%d-%H.%M.%S')
            temp_file = os.path.join(output_dir, f'tmp_{timestamp}.npz')
        
        # Initialize population
        self.disp_param.initialize_population(self.mutation_proba)
        
        # Main optimization loop
        while (self.current_generation < self.max_iter and 
               not self._check_convergence()):
            
            print(f"\n=== Generation {self.current_generation + 1}/{self.max_iter} ===")
            
            # Store current parameters
            self.disp_param_storage.append(self.disp_param.param_table.copy())
            
            # Evaluate generation
            start_time = datetime.now()
            fitness_scores = self._evaluate_generation(self.current_generation)
            end_time = datetime.now()
            
            print(f"Generation evaluation time: {(end_time - start_time).total_seconds():.2f}s")
            
            # Store fitness scores
            self.fitness_storage[:, self.current_generation] = fitness_scores
            
            # Print statistics
            best_fitness = np.min(fitness_scores)
            mean_fitness = np.mean(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            
            print(f"Best fitness: {best_fitness:.6e}")
            print(f"Mean fitness: {mean_fitness:.6e}")
            
            # Check for early termination
            if self.testing_mode:
                print("Testing mode - stopping after one generation")
                break
                
            if best_fitness < self.fitness_tol:
                print(f"Convergence achieved! Fitness {best_fitness} < {self.fitness_tol}")
                break
            
            # Compute population diversity
            param_arrays = {
                param_name: self.disp_param.param_table[param_name].values
                for param_name in self.disp_param.param_settings.keys()
            }
            diversity = self.population_manager.compute_diversity(param_arrays)
            self.diversity_history.append(diversity)
            print(f"Population diversity: {diversity:.4f}")
            
            # Adaptive mutation rate
            adaptive_rate = max(0.01, self.mutation_proba * np.tanh(3 * (1 - self.current_generation / self.max_iter)))
            print(f"Mutation rate: {adaptive_rate:.4f}")
            
            # Create new generation
            self.disp_param.mutate_pop(fitness_scores, adaptive_rate)
            
            self.current_generation += 1
            
            # Save temporary results
            if save_data:
                self._save_temp_results(temp_file)
        
        print(f"\nOptimization completed after {self.current_generation} generations")
        
        # Prepare final results
        results = self._compile_results()
        
        # Save final results
        if save_data:
            final_file = os.path.join(output_dir, f'{timestamp}_Genetic_evolution.npz')
            self._save_final_results(final_file, results)
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return results
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged.""" 
        if self.current_generation == 0:
            return False
            
        # Match MATLAB logic: ~any(fitnessArray(:,max(1,i_gen-1))<fitnessTol)
        gen_idx = max(0, self.current_generation - 1)
        current_gen_fitness = self.fitness_storage[:, gen_idx]
        
        # Check if any individual in current generation has fitness below threshold
        converged_individuals = current_gen_fitness < self.fitness_tol
        has_converged = np.any(converged_individuals)
        
        return has_converged
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile optimization results."""
        best_gen = self.current_generation - 1 if self.current_generation > 0 else 0
        best_idx = np.argmin(self.fitness_storage[:, best_gen])
        
        results = {
            'dispersion_params': self.disp_param_storage,
            'psi_evolution': self.psi_storage[:, :, :, :self.current_generation],
            'fitness_evolution': self.fitness_storage[:, :self.current_generation], 
            'variance_gl': self.var_gl_storage[:, :self.current_generation],
            'convergence_flags': self.conv_flag_storage[:, :self.current_generation],
            'best_fitness_history': np.array(self.best_fitness_history),
            'diversity_history': np.array(self.diversity_history),
            'final_generation': self.current_generation,
            'best_individual_idx': best_idx,
            'best_fitness': np.min(self.fitness_storage[:, best_gen]),
            'optimization_params': {
                'n_pop': self.n_pop,
                'max_iter': self.max_iter,
                'fitness_tol': self.fitness_tol,
                'mutation_proba': self.mutation_proba,
                'n_modes': self.n_modes
            }
        }
        
        return results
    
    def _save_temp_results(self, filename: str):
        """Save temporary results during optimization."""
        current_results = self._compile_results()
        np.savez_compressed(filename, **current_results)
    
    def _save_final_results(self, filename: str, results: Dict[str, Any]):
        """Save final optimization results."""
        np.savez_compressed(filename, **results)
        print(f"Results saved to: {filename}")
    
    def get_best_individual(self) -> Tuple[int, Dict[str, Any]]:
        """
        Get the best individual from the final generation.
        
        Returns
        -------
        tuple
            (best_index, best_parameters)
        """
        if self.current_generation == 0 and np.all(self.fitness_storage == self.fitness_tol):
            raise ValueError("No optimization has been run yet")
            
        best_gen = max(0, self.current_generation - 1)
        best_idx = np.argmin(self.fitness_storage[:, best_gen])
        
        # Extract best parameters
        best_params = {}
        for param_name in self.disp_param.param_settings.keys():
            best_params[param_name] = self.disp_param.param_table.loc[best_idx, param_name]
        
        best_params['fitness'] = self.fitness_storage[best_idx, best_gen]
        best_params['psi_final'] = self.psi_storage[:, :, best_idx, best_gen]
        
        return best_idx, best_params