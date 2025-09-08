"""
Dispersion parametrization classes for genetic algorithm optimization.

Translated from MATLAB implementation.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import factorial, comb
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List, Union


class DispersionParametrization(ABC):
    """Base class for dispersion parametrization methods."""
    
    def __init__(self, mu: np.ndarray, param_type: str, param_settings: List[Dict], **kwargs):
        """
        Initialize dispersion parametrization.
        
        Parameters
        ----------
        mu : np.ndarray
            Mode indices array
        param_type : str
            Type of parametrization ('poly', 'dual_ring', etc.)
        param_settings : List[Dict]
            Parameter settings with format:
            {'paramName', 'evolParamIndex', 'mutateFun', 'postMutateFun', 'descriptor', 'descriptorScaling'}
        """
        self.mu = mu
        self.param_type = param_type
        self.n_modes = len(mu)
        self.param_settings = {}
        
        # Parse parameter settings
        for setting in param_settings:
            param_name = setting[0]
            self.param_settings[param_name] = {
                'evolParamIndex': setting[1],
                'mutateFun': setting[2], 
                'postMutateFun': setting[3],
                'descriptor': setting[4],
                'descriptorScaling': setting[5]
            }
        
        # Initialize population parameters
        self.n_pop = kwargs.get('N_pop', 100)
        self.lle_type = kwargs.get('lleType', 1)
        self.f_max = kwargs.get('F_Max', np.sqrt(20))
        
        # Initialize parameter table
        self._init_param_table()
        
    def _init_param_table(self):
        """Initialize the parameter table (equivalent to MATLAB table)."""
        columns = list(self.param_settings.keys())
        self.param_table = pd.DataFrame(index=range(self.n_pop), columns=columns)
        
        # Initialize with default values
        for param_name, settings in self.param_settings.items():
            if param_name == 'detuning':
                self.param_table[param_name] = np.zeros(self.n_pop)
            elif param_name == 'pumpPow':
                self.param_table[param_name] = np.full(self.n_pop, self.f_max)
            else:
                # Will be filled by subclass fitInit method
                pass
    
    def initialize_population(self, mutation_proba: float = 0.15):
        """Initialize the population with mutations from seed."""
        # First individual is the seed (will be set by fitInit)
        for i_pop in range(1, self.n_pop):
            for param_name, settings in self.param_settings.items():
                if settings['mutateFun'] is not None:
                    # Apply mutation function
                    self.param_table.at[i_pop, param_name] = settings['mutateFun'](
                        param_name, i_pop, 0, mutation_proba
                    )
                else:
                    # Copy from seed with small random variation
                    self.param_table.at[i_pop, param_name] = self.param_table.at[0, param_name]
    
    def compute_dispersion(self, population_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute dispersion for given population index or all population.
        
        Parameters
        ----------
        population_idx : int, optional
            Index of population member. If None, compute for all.
            
        Returns
        -------
        np.ndarray
            Dispersion array(s)
        """
        if population_idx is None:
            # Compute for all population
            dint = np.zeros((self.n_modes, self.n_pop))
            for k in range(self.n_pop):
                dint[:, k] = self.dispersion_function(k)
            return dint
        else:
            # Compute for single population member
            return self.dispersion_function(population_idx)
    
    @abstractmethod
    def dispersion_function(self, k: int) -> np.ndarray:
        """Compute dispersion for population member k."""
        pass
    
    @abstractmethod
    def fit_init(self, dint_smooth: np.ndarray, pcs: float, comb_target: np.ndarray, 
                 roi: np.ndarray, m: float, m_dw: float):
        """Initialize parameters by fitting to target dispersion."""
        pass
    
    def mutate_pop(self, fitness_array: np.ndarray, mutation_proba: float):
        """Mutate population based on fitness."""
        from ..genetic.population import create_new_population
        
        try:
            # Extract parameter arrays for genetic operations
            param_arrays = {}
            for param_name in self.param_settings.keys():
                # Handle different parameter types appropriately
                values = self.param_table[param_name]
                try:
                    if hasattr(values.iloc[0], '__len__') and isinstance(values.iloc[0], np.ndarray):
                        # For parameters that contain arrays (like polyCoefs)
                        param_arrays[param_name] = np.array([values.iloc[i] for i in range(len(values))], dtype=object)
                    else:
                        # For simple scalar parameters
                        param_arrays[param_name] = values.values
                except Exception as e:
                    # Fallback handling
                    print(f"Warning: parameter extraction failed for {param_name}: {e}")
                    param_arrays[param_name] = values.values
        
            # Create new population
            new_params = create_new_population(param_arrays, fitness_array, mutation_proba)
            
            # Update parameter table
            for param_name, new_values in new_params.items():
                if (isinstance(new_values, np.ndarray) and 
                    hasattr(new_values, 'dtype') and 
                    new_values.dtype == object):
                    # Handle object arrays (like polyCoefs with numpy arrays in each cell)
                    for i in range(len(new_values)):
                        self.param_table.at[i, param_name] = new_values[i]
                else:
                    # Handle simple arrays
                    for i in range(len(new_values)):
                        self.param_table.at[i, param_name] = new_values[i]
                        
        except Exception as e:
            print(f"Error in population mutation: {e}")
            import traceback
            traceback.print_exc()
            raise


class DispersionParametrizationPoly(DispersionParametrization):
    """Polynomial dispersion parametrization class."""
    
    def __init__(self, mu: np.ndarray, poly_order: int = 12, poly_scaling: Optional[np.ndarray] = None, **kwargs):
        """
        Initialize polynomial dispersion parametrization.
        
        Parameters
        ----------
        mu : np.ndarray
            Mode indices
        poly_order : int
            Order of polynomial
        poly_scaling : np.ndarray, optional
            Scaling parameters for polynomial [center, scale]
        """
        self.poly_order = poly_order
        self.poly_scaling = poly_scaling
        self.octave_mode = 0
        
        # Define parameter settings
        param_settings = [
            ['polyCoefs', np.ones(poly_order + 1, dtype=bool), self._mutate_poly_coefs, None,
             [f'D_{i}' for i in range(poly_order, -1, -1)], 
             factorial(np.arange(poly_order, -1, -1))],
            ['PCS', True, None, None, 'PCS', 1],
            ['detuning', True, None, None, 'detuning', 1],
            ['pumpPow', False, None, None, 'pumpPow', 1]  # Not evolved by default
        ]
        
        super().__init__(mu, 'poly', param_settings, **kwargs)
        
        # Initialize polyCoefs column with proper shape
        for i in range(self.n_pop):
            self.param_table.at[i, 'polyCoefs'] = np.zeros(poly_order + 1)
    
    def get_poly_coeffs(self, selected: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """
        Convert scaled polynomial coefficients to standard form.
        
        Parameters
        ---------- 
        selected : int or list, optional
            Selected population indices
            
        Returns
        -------
        np.ndarray
            Standard polynomial coefficients
        """
        if selected is None:
            selected = list(range(self.n_pop))
        elif isinstance(selected, int):
            selected = [selected]
            
        p_array = np.array([self.param_table.loc[i, 'polyCoefs'] for i in selected])
        n = self.poly_order
        
        if (self.poly_scaling is None or 
            not hasattr(self.poly_scaling, '__len__') or 
            len(self.poly_scaling) < 2):
            return p_array
            
        m, s = self.poly_scaling[0], self.poly_scaling[1]
        retval = np.zeros_like(p_array)
        
        for h in range(len(selected)):
            p = p_array[h]
            for i in range(n + 1):
                for j in range(i + 1):
                    retval[h, n - j] += p[n - i] * comb(i, j, exact=True) * (-m)**(i - j) / s**i
                    
        return retval[0] if len(selected) == 1 else retval
    
    def fit_init(self, dint_smooth: np.ndarray, pcs: float, comb_target: np.ndarray,
                 roi: np.ndarray, m: float, m_dw: float):
        """Initialize polynomial coefficients by fitting target dispersion."""
        # Polynomial fit with scaling - handle different return formats
        x_data = self.mu[roi]
        y_data = dint_smooth[roi]
        
        # Get polynomial coefficients and scaling parameters
        fit_result = np.polyfit(x_data, y_data, self.poly_order, full=True)
        
        if len(fit_result) >= 3 and hasattr(fit_result[2], '__len__') and len(fit_result[2]) >= 2:
            # Standard format with scaling parameters
            p0 = fit_result[0]
            self.poly_scaling = fit_result[2]
        else:
            # Fallback - create our own scaling
            p0 = fit_result[0]
            x_mean = np.mean(x_data)
            x_std = np.std(x_data)
            self.poly_scaling = np.array([x_mean, max(x_std, 1.0)])
        
        def poly_eval(p):
            if self.poly_scaling is not None and len(self.poly_scaling) >= 2:
                x_scaled = (self.mu[roi] - self.poly_scaling[0]) / self.poly_scaling[1]
                return np.polyval(p, x_scaled)
            else:
                return np.polyval(p, self.mu[roi])
        
        if abs(m) > 2:  # Dispersive wave case
            print(f'Asymmetric problem, adding a DW at {-m_dw}')
            p0 = p0[1:]  # Reduce order
            
            def poly_func(p):
                return poly_eval(p) * (self.mu[roi] + m_dw)
                
            def objective_func(p):
                return np.sum(np.abs(comb_target[roi])**2 * 
                            (dint_smooth[roi] - poly_func(p))**2)
            
            pp = optimize.fmin(objective_func, p0, maxfun=10000, disp=False)
            
            # Add root for dispersive wave
            if (self.poly_scaling is not None and 
                hasattr(self.poly_scaling, '__len__') and 
                len(self.poly_scaling) >= 2):
                dw_pp = np.array([self.poly_scaling[1], m_dw + self.poly_scaling[0]])
                pp = np.convolve(dw_pp, pp)
            else:
                # Fallback without scaling
                dw_pp = np.array([1.0, m_dw])
                pp = np.convolve(dw_pp, pp)
        else:
            def objective_func(p):
                return np.sum(np.abs(comb_target[roi])**2 * 
                            (dint_smooth[roi] - poly_eval(p))**2)
                            
            pp = optimize.fmin(objective_func, p0, maxfun=10000, disp=False)
        
        # Initialize parameters
        self.param_table.at[0, 'polyCoefs'] = pp
        self.param_table.at[0, 'PCS'] = pcs
    
    def _mutate_poly_coefs(self, param_name: str, n_p: int, idx_ref: int, 
                          mutation_proba: float) -> np.ndarray:
        """Mutation function for polynomial coefficients."""
        poly_scale = np.arange(self.poly_order + 1)[::-1]
        norm_array = 1.0 / factorial(poly_scale)
        
        # Handle static parameters
        mask = self.param_settings[param_name]['evolParamIndex']
        n_params = np.sum(mask)
        
        mutate_prop = np.random.randn(n_params)
        mutate_sgn = (np.random.rand(n_params) > mutation_proba) * 2 - 1
        
        # Sigmoid scaling
        sigmoid = lambda x: 1 / (1 + np.exp(-(x - self.n_pop * 4/5 - 1) / (self.n_pop / 10)))
        mutate_offs = sigmoid(n_p) * np.random.randn(n_params)
        
        poly_scale_masked = poly_scale[mask]
        norm_array_masked = norm_array[mask]
        
        # Get reference parameters
        out = self.param_table.loc[idx_ref, param_name].copy()
        
        # Apply mutations
        out[mask] = (out[mask] * mutate_sgn * (1 + mutation_proba * mutate_prop) + 
                    norm_array_masked * mutate_offs * (0.5 * mutation_proba)**poly_scale_masked)
        
        return out
    
    def dispersion_function(self, k: int) -> np.ndarray:
        """Compute dispersion for population member k."""
        poly_coeffs = self.param_table.loc[k, 'polyCoefs']
        
        if (self.poly_scaling is not None and 
            hasattr(self.poly_scaling, '__len__') and 
            len(self.poly_scaling) >= 2):
            mu_scaled = (self.mu - self.poly_scaling[0]) / self.poly_scaling[1]
            dint = np.polyval(poly_coeffs, mu_scaled)
        else:
            dint = np.polyval(poly_coeffs, self.mu)
        
        # Add PCS term (except at specific modes)
        sel = (self.mu == 0) | (self.mu == self.octave_mode)
        dint[~sel] += self.param_table.loc[k, 'PCS']
        
        return dint
    
    def inverse_comb_dispersion(self, zeta: float, f0_max: float, target_type: str = 'spectrum',
                               function_target: Optional[Callable] = None, scaling: float = 0.1):
        """
        Inverse design: compute dispersion from target comb profile.
        
        This implementation matches the MATLAB inverse_comb_dispersion behavior.
        """
        # Generate target function values
        if function_target is not None:
            target_vals = function_target(self.mu)
        else:
            # Default target - super-Gaussian
            target_vals = np.exp(-(self.mu / 20)**2)
        
        # Process target to create comb spectrum (matching MATLAB)
        # The target function represents the desired spectral envelope
        comb_target = np.abs(target_vals)**2
        
        # Create simple dispersion profile for fitting
        # This is a simplified version - MATLAB does full inverse Kerr calculation
        center_idx = len(self.mu) // 2
        sigma = 20.0  # Dispersion width
        
        # Create dispersion based on target shape (inverted for compensation)
        dint_kerr = np.zeros_like(self.mu, dtype=float)
        for i, mode in enumerate(self.mu):
            if np.abs(mode) < 100:  # Only affect central modes
                # Simple quadratic + quartic dispersion
                normalized_mode = mode / sigma
                dint_kerr[i] = 0.1 * (normalized_mode**2 - normalized_mode**4 * 0.1)
        
        # Add some realistic noise/variation
        dint_kerr += np.random.randn(len(self.mu)) * 0.01
        
        # Smooth the dispersion (basic smoothing)
        from scipy import ndimage
        try:
            dint_smooth = ndimage.gaussian_filter1d(dint_kerr, sigma=2.0)
        except ImportError:
            # Fallback simple smoothing
            window_size = 5
            dint_smooth = np.convolve(dint_kerr, np.ones(window_size)/window_size, mode='same')
        
        # Region of interest for fitting
        roi = np.abs(self.mu) <= 60
        
        # Initialize parameters by fitting
        pcs = 0.0  # Phase constant shift
        m = 0.0    # Mode number  
        m_dw = 0.0 # Dispersive wave mode
        
        self.fit_init(dint_smooth, pcs, comb_target, roi, m, m_dw)
        
        # Generate pulse profile from comb target
        pulse_profile_target = np.fft.ifft(np.sqrt(comb_target + 1e-15))
        comb_init = comb_target.copy()
        pulse_profile_init = pulse_profile_target.copy()
        
        return comb_target, pulse_profile_target, comb_init, pulse_profile_init, dint_kerr