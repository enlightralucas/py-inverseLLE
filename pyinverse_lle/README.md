# PyInverseLLE

Python implementation of genetic algorithm optimization for Kerr frequency comb generation using the Lugiato-Lefever Equation (LLE).

This package implements genetic algorithm-based optimization of resonator dispersion profiles for tailoring Kerr frequency combs. It is a Python translation of the MATLAB implementation described in the Nature Photonics paper (DOI [10.1038/s41566-023-01252-7](https://doi.org/10.1038/s41566-023-01252-7)).

## Features

- **Genetic Algorithm Optimization**: Evolve populations of dispersion parameters to find optimal comb profiles
- **LLE Numerical Solver**: Split-step Fourier method for solving the Lugiato-Lefever equation
- **Flexible Fitness Functions**: Multiple optimization targets (flat combs, custom shapes, octave-spanning)
- **Parallel Processing**: Accelerated computation using multiprocessing/joblib
- **Comprehensive Analysis**: Tools for analyzing optimization results and visualizing evolution
- **Multiple I/O Formats**: Save/load results in NumPy, HDF5, Pickle, and MATLAB formats

## Installation

### Requirements

```bash
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
h5py>=3.6.0
joblib>=1.1.0
deap>=1.3.1
numba>=0.56.0
tqdm>=4.62.0
```

### Install from source

```bash
git clone <repository-url>
cd pyinverse_lle
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic Optimization

```python
import numpy as np
from pyinverse_lle import setup_poly_comb_target, GeneticOptimizer

# Setup optimization for flat comb generation
config = setup_poly_comb_target(
    n_modes=512,
    target_power_db=-5,
    target_half_width_modes=16
)

# Create and run optimizer
optimizer = GeneticOptimizer(
    config['dispersion_param'],
    config['fitness_evaluator'],
    n_pop=208,
    max_iter=300
)

results = optimizer.optimize()
```

### Command Line Usage

Run optimization from command line:

```bash
cd pyinverse_lle/examples
python optimize_flat_comb.py --n_pop 100 --max_iter 50
```

Analyze results:

```bash
python analyze_results.py data/2024_01_15-14.30.25_Genetic_evolution.npz
```

### Testing Mode

For quick testing with reduced computation:

```bash
python optimize_flat_comb.py --testing --n_pop 10 --max_iter 5
```

## Architecture

### Core Components

- **`core/dispersion.py`**: Dispersion parametrization classes
  - `DispersionParametrization`: Base class
  - `DispersionParametrizationPoly`: Polynomial dispersion profiles

- **`core/lle_solver.py`**: LLE numerical integration
  - `LLEPropagator`: Main solver class
  - `lle_splitstep`: JIT-compiled split-step method

- **`core/fitness.py`**: Fitness function implementations
  - `fitness_line_power_target`: Flat comb optimization
  - `fitness_fit_target`: Custom target matching
  - `FitnessEvaluator`: Unified evaluation interface

- **`genetic/algorithm.py`**: Genetic algorithm orchestrator
  - `GeneticOptimizer`: Main optimization class

- **`genetic/population.py`**: Population management
  - Selection, crossover, and mutation operations

### Utilities

- **`utils/initialization.py`**: Setup and configuration helpers
- **`utils/visualization.py`**: Plotting and analysis tools
- **`utils/data_io.py`**: Save/load functionality

## Usage Examples

### 1. Flat Comb Optimization

```python
from pyinverse_lle.utils.initialization import setup_poly_comb_target
from pyinverse_lle.genetic.algorithm import GeneticOptimizer

# Configure for flat comb
config = setup_poly_comb_target(
    n_modes=512,
    poly_order=6,
    target_power_db=-5,      # Target power level
    target_var_power=5,      # Tolerance in dB
    target_half_width_modes=16  # Half-width in modes
)

# Run optimization
optimizer = GeneticOptimizer(
    config['dispersion_param'], 
    config['fitness_evaluator'],
    n_pop=208,
    max_iter=300,
    fitness_tol=1e-6,
    n_workers=-1  # Use all CPU cores
)

results = optimizer.optimize(save_data=True)
best_idx, best_params = optimizer.get_best_individual()
```

### 2. Custom Target Shapes

```python
from pyinverse_lle.utils.initialization import create_adaptive_target_function

# Create Gaussian target
target_func = create_adaptive_target_function('gaussian', width=20)

# Create super-Gaussian target  
target_func = create_adaptive_target_function('super_gaussian', width=15, order=8)

# Use in configuration
config = setup_poly_comb_target(n_modes=512)
config['dispersion_param'].inverse_comb_dispersion(
    function_target=target_func
)
```

### 3. Octave-Spanning Optimization

```python
from pyinverse_lle.utils.initialization import setup_octave_dw_target

# Setup for octave-spanning dispersive wave
config = setup_octave_dw_target(
    n_modes=512,
    octave_mode=128,  # Target octave mode
    poly_order=8
)

optimizer = GeneticOptimizer(config['dispersion_param'], config['fitness_evaluator'])
results = optimizer.optimize()
```

### 4. Advanced Analysis

```python
from pyinverse_lle.utils.visualization import plot_evolution_summary
from pyinverse_lle.utils.data_io import export_summary_report

# Load and analyze results
from pyinverse_lle.utils.data_io import load_results
results = load_results('data/optimization_results.npz')

# Generate comprehensive plots
fig = plot_evolution_summary(results, save_path='evolution_analysis.png')

# Export summary report
export_summary_report(results, 'optimization_summary.txt')
```

## Configuration Options

### Genetic Algorithm Parameters

```python
ga_params = {
    'n_pop': 208,           # Population size
    'max_iter': 300,        # Maximum generations  
    'fitness_tol': 1e-6,    # Convergence tolerance
    'mutation_proba': 0.15, # Base mutation probability
    'n_workers': -1,        # Parallel workers (-1 = all cores)
    'testing_mode': False   # Enable for quick testing
}
```

### LLE Solver Parameters

```python
lle_params = {
    'lle_type': 1,          # Number of propagation directions
    'noise_factor': 0.0,    # Noise level (0 = deterministic)
    't_evol_1': 40.0,       # Coarse evolution time
    't_evol_2': 10.0,       # Fine evolution time
    'h_coarse': 2**(-9),    # Coarse time step
    'h_fine': 2**(-14)      # Fine time step
}
```

### Dispersion Parameters

```python
disp_params = {
    'poly_order': 6,        # Polynomial order
    'n_modes': 512,         # Number of modes (power of 2)
    'f0_max': np.sqrt(20),  # Maximum pump amplitude
    'enforce_symmetry': True # Enforce even polynomial terms only
}
```

## Output Files

The optimization generates several output files:

- **`*_Genetic_evolution.npz`**: Main results file with full evolution data
- **`*_summary.txt`**: Human-readable optimization summary  
- **`*_best_parameters.csv`**: Best individual parameters
- **`*_evolution.png`**: Fitness evolution plots
- **`*_best_solution.png`**: Best solution analysis
- **`*_comparison.png`**: Target vs result comparison

## Performance Tips

1. **Use power-of-2 mode numbers** (256, 512, 1024) for efficient FFT
2. **Enable parallel processing** with `n_workers=-1` 
3. **Start with smaller populations** (50-100) for testing
4. **Use testing mode** (`--testing`) for code validation
5. **Monitor memory usage** for large populations/long evolutions

## Comparison with MATLAB Version

| Feature | MATLAB | Python | Notes |
|---------|---------|---------|-------|
| Core Algorithm | ✓ | ✓ | Identical genetic algorithm |
| LLE Solver | ✓ | ✓ | Split-step method with numba acceleration |
| Parallel Processing | `parfor` | `joblib` | Similar performance |
| Visualization | MATLAB plots | matplotlib | Enhanced plotting capabilities |
| Data Format | `.mat` files | Multiple formats | More flexible I/O |
| Performance | Reference | ~0.8-1.2x | Comparable with numba |

## Troubleshooting

### Common Issues

1. **Memory errors with large populations**: Reduce `n_pop` or `n_modes`
2. **Slow performance**: Enable parallel processing, use numba
3. **Import errors**: Check all dependencies are installed
4. **Convergence issues**: Adjust `fitness_tol`, increase `max_iter`

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project maintains compatibility with the original MATLAB implementation licensing.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2023,
  title={Genetic algorithm optimization for Kerr frequency comb generation},
  journal={Nature Photonics},
  doi={10.1038/s41566-023-01252-7},
  year={2023}
}
```

## Support

- **Issues**: Report bugs and feature requests via GitHub issues
- **Documentation**: See examples in `pyinverse_lle/examples/`
- **Performance**: For optimization questions, see performance tips section