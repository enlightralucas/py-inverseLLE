"""
Lugiato-Lefever Equation (LLE) numerical solvers.

Implements split-step Fourier method for solving the LLE.
Translated from MATLAB implementation.
"""

import numpy as np
from typing import Optional, Callable, Union, Tuple
from numba import jit


def lle_splitstep(psi: np.ndarray, d_hat: np.ndarray, s_vec: np.ndarray, 
                  gamma_l: float, h: float) -> np.ndarray:
    """
    Split-step method for LLE propagation.
    
    Note: Removed @jit decorator due to numba compatibility issues with numpy.fft
    
    Parameters
    ----------
    psi : np.ndarray
        Input field in time domain
    d_hat : np.ndarray
        Dispersion operator (pre-computed)
    s_vec : np.ndarray
        Pump function in frequency domain
    gamma_l : float
        Nonlinear parameter (usually 1)
    h : float
        Step size
        
    Returns
    -------
    np.ndarray
        Output field after one step
    """
    # Nonlinear step (time domain)
    # Kerr nonlinearity + pump injection
    psi = np.exp(h * 1j * gamma_l * np.abs(psi)**2) * (psi + s_vec * h)
    
    # Transform to frequency domain
    psi = np.fft.fft(psi)
    
    # Linear step (frequency domain) - dispersion
    psi = d_hat * psi
    
    # Transform back to time domain
    psi = np.fft.ifft(psi)
    
    return psi


class LLEPropagator:
    """
    Lugiato-Lefever Equation propagator using split-step Fourier method.
    """
    
    def __init__(self, n_modes: int):
        """
        Initialize LLE propagator.
        
        Parameters
        ----------
        n_modes : int
            Number of modes/grid points
        """
        self.n_modes = n_modes
        self.f_mask = self._create_pump_mask()
        
    def _create_pump_mask(self) -> np.ndarray:
        """Create pump mask for central mode excitation."""
        f_mask = np.zeros(self.n_modes)
        f_mask[0] = 1  # Central mode
        return np.fft.fft(f_mask)
    
    def _stop_condition(self, t_current: float, t_evol: float, psi: np.ndarray, 
                       psi_old: np.ndarray, store_final_only: bool) -> bool:
        """Check stopping conditions for evolution."""
        if store_final_only:
            if t_current == 0:
                return False
            else:
                return (t_current >= t_evol or 
                       np.linalg.norm(psi - psi_old) < 1e-10)
        else:
            return t_current >= t_evol
    
    def _gen_light_metric(self, psi: np.ndarray) -> float:
        """Compute generated light metric for stability assessment."""
        return np.sum(np.abs(psi - np.mean(psi))**2)
    
    def propagate(self, psi_init: np.ndarray, f_pump: Union[float, Callable], 
                  zeta: Union[float, Callable], dispersion: np.ndarray, 
                  noise_factor: float, t_evol: float, 
                  time_step: Union[float, Callable], n_store: int,
                  coupling: Optional[float] = None, 
                  xpm_bidir: bool = True,
                  store_all_evo: bool = False,
                  report_option: str = 'none') -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Propagate the LLE using split-step method.
        
        Parameters
        ----------
        psi_init : np.ndarray
            Initial field state
        f_pump : float or callable
            Pump amplitude (constant or function of time)
        zeta : float or callable  
            Detuning parameter (constant or function of time)
        dispersion : np.ndarray
            Dispersion profile D (complex, includes loss and dispersion)
        noise_factor : float
            Noise level
        t_evol : float
            Evolution time
        time_step : float or callable
            Time step (constant or adaptive)
        n_store : int
            Number of storage points
        coupling : float, optional
            Coupling coefficient (not implemented)
        xpm_bidir : bool
            Enable bidirectional XPM (not implemented)
        store_all_evo : bool
            Store full evolution or final state only
        report_option : str
            Reporting option ('none', 'plot', 'waitbar')
            
        Returns
        -------
        tuple
            (final_psi, final_time, variance_gen_light)
        """
        # Validate inputs
        n_modes, n_dir = psi_init.shape if psi_init.ndim > 1 else (len(psi_init), 1)
        psi = psi_init.copy()
        
        if psi.ndim == 1:
            psi = psi.reshape(-1, 1)
            n_dir = 1
        
        # Handle variable parameters
        is_tstep_variable = callable(time_step)
        is_zeta_variable = callable(zeta)
        is_f_variable = callable(f_pump)
        
        # Initialize operators
        d_hat = None
        
        def update_operators(h_val):
            nonlocal d_hat
            d_hat = np.exp(h_val * dispersion)
        
        # Initial step setup
        h = time_step(0) if is_tstep_variable else time_step
        update_operators(h)
        
        # Initialize detuning and pump
        z_val = zeta(0) if is_zeta_variable else zeta
        f_val = f_pump(0) if is_f_variable else f_pump
        
        # Pump vector setup
        if np.isscalar(f_val):
            f_vec = self.f_mask * f_val
        else:
            f_vec = self.f_mask * f_val[0] if hasattr(f_val, '__len__') else self.f_mask * f_val
        
        # Storage setup
        t_store_flags = np.linspace(0, t_evol, n_store)
        store_final_only = not (is_zeta_variable or is_f_variable or store_all_evo)
        
        # Generated light tracking
        gen_light = np.zeros((n_store, n_dir))
        gen_light[0, :] = self._gen_light_metric(psi)
        
        if not store_final_only:
            psi_storage = np.zeros((n_modes, n_dir, n_store), dtype=complex)
            t_storage = np.zeros(n_store)
            psi_storage[:, :, 0] = psi
            t_storage[0] = 0
        
        # Main evolution loop
        psi_old = psi.copy()
        t_current = 0.0
        j = 1
        
        while not self._stop_condition(t_current, t_evol, psi, psi_old, store_final_only):
            t_current += h
            
            # Update variable parameters
            if is_zeta_variable:
                z_val = zeta(t_current)
            if is_f_variable:
                f_val = f_pump(t_current)
                f_vec = self.f_mask * f_val
            
            # Store previous state
            psi_old = psi.copy()
            
            # Apply detuning to dispersion operator
            d_hat_with_detuning = np.exp(-1j * z_val * h) * d_hat
            
            # Split-step propagation for each direction
            for i_dir in range(n_dir):
                psi[:, i_dir] = lle_splitstep(
                    psi[:, i_dir], d_hat_with_detuning, f_vec, 1.0, h
                )
            
            # Storage and updates
            if j < len(t_store_flags) and t_current >= t_store_flags[j]:
                gen_light[j, :] = self._gen_light_metric(psi)
                
                if not store_final_only:
                    psi_storage[:, :, j] = psi
                    t_storage[j] = t_current
                
                # Update time step if variable
                if is_tstep_variable:
                    h = time_step(t_current)
                    update_operators(h)
                
                j += 1
            
            # Add noise
            if noise_factor != 0:
                noise_real = noise_factor * h * np.random.randn(*psi.shape)
                noise_imag = noise_factor * h * np.random.randn(*psi.shape)
                noise = noise_real + 1j * noise_imag
                psi += noise
        
        # Final processing
        psi = (psi + psi_old) / 2  # Average last two steps
        
        # Compute stability metric
        var_gen_l = np.std(gen_light[n_store//2:, :], axis=0) / np.mean(gen_light[n_store//2:, :], axis=0)
        
        if store_final_only:
            return psi.squeeze(), t_current, var_gen_l
        else:
            return psi_storage, t_storage, var_gen_l


def spectrum_f(x: np.ndarray) -> np.ndarray:
    """
    Compute power spectrum from field.
    
    Parameters
    ----------
    x : np.ndarray
        Input field
        
    Returns
    -------
    np.ndarray
        Power spectrum
    """
    return np.fft.ifftshift(np.abs(np.fft.ifft(x, axis=0))**2, axes=0)


class AdaptiveTimeStep:
    """Helper class for adaptive time stepping."""
    
    def __init__(self, h_large: float, h_fine: float, transition_param: float = 15, 
                 transition_point: float = 0.4):
        """
        Initialize adaptive time step.
        
        Parameters
        ----------
        h_large : float
            Large time step
        h_fine : float
            Fine time step  
        transition_param : float
            Transition sharpness parameter
        transition_point : float
            Relative transition point (0-1)
        """
        self.h_large = h_large
        self.h_fine = h_fine
        self.transition_param = transition_param
        self.transition_point = transition_point
    
    def __call__(self, t: float, t_evol: float) -> float:
        """
        Compute adaptive time step.
        
        Parameters
        ----------
        t : float
            Current time
        t_evol : float
            Total evolution time
            
        Returns
        -------
        float
            Time step
        """
        # Hyperbolic tangent transition from large to fine step
        progress = t / t_evol
        transition = (1 + np.tanh(self.transition_param * 
                                (self.transition_point - progress))) / 2
        return (self.h_large - self.h_fine) * transition + self.h_fine