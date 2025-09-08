"""
PyInverseLLE: Python implementation of genetic algorithm optimization for Kerr frequency comb generation

This package implements genetic algorithm-based optimization of resonator dispersion 
profiles for tailoring Kerr frequency combs using the Lugiato-Lefever Equation (LLE).

Based on the MATLAB implementation described in:
Nature Photonics paper (DOI 10.1038/s41566-023-01252-7)
"""

__version__ = "0.1.0"
__author__ = "Converted from MATLAB implementation"

from .core.dispersion import DispersionParametrization, DispersionParametrizationPoly
from .core.lle_solver import LLEPropagator
from .core.fitness import fitness_line_power_target, fitness_fit_target
from .genetic.algorithm import GeneticOptimizer

__all__ = [
    'DispersionParametrization',
    'DispersionParametrizationPoly', 
    'LLEPropagator',
    'fitness_line_power_target',
    'fitness_fit_target',
    'GeneticOptimizer'
]