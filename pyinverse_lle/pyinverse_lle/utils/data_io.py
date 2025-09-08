"""
Data I/O utilities for saving and loading optimization results.

Provides functions for saving results in various formats.
"""

import numpy as np
import pandas as pd
import h5py
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import os


def save_results(results: Dict[str, Any], filepath: str, 
                format_type: str = 'npz') -> None:
    """
    Save optimization results in specified format.
    
    Parameters
    ----------
    results : dict
        Results dictionary from GeneticOptimizer
    filepath : str
        Output file path
    format_type : str
        Format type ('npz', 'hdf5', 'pickle', 'matlab')
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', 
                exist_ok=True)
    
    if format_type == 'npz':
        save_results_npz(results, filepath)
    elif format_type == 'hdf5':
        save_results_hdf5(results, filepath)
    elif format_type == 'pickle':
        save_results_pickle(results, filepath)
    elif format_type == 'matlab':
        save_results_matlab(results, filepath)
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    print(f"Results saved to: {filepath}")


def save_results_npz(results: Dict[str, Any], filepath: str) -> None:
    """Save results in NumPy compressed format."""
    # Convert pandas DataFrames to arrays if present
    save_dict = {}
    
    for key, value in results.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], pd.DataFrame):
                # Convert list of DataFrames to structured format
                save_dict[f'{key}_list'] = [df.to_dict() for df in value]
            else:
                save_dict[key] = np.array(value) if not isinstance(value[0], dict) else value
        elif isinstance(value, pd.DataFrame):
            save_dict[f'{key}_data'] = value.values
            save_dict[f'{key}_columns'] = value.columns.tolist()
            save_dict[f'{key}_index'] = value.index.tolist()
        else:
            save_dict[key] = value
    
    # Add metadata
    save_dict['_metadata'] = {
        'save_time': datetime.now().isoformat(),
        'version': '0.1.0',
        'format': 'npz'
    }
    
    np.savez_compressed(filepath, **save_dict)


def save_results_hdf5(results: Dict[str, Any], filepath: str) -> None:
    """Save results in HDF5 format."""
    with h5py.File(filepath, 'w') as f:
        # Add metadata
        f.attrs['save_time'] = datetime.now().isoformat()
        f.attrs['version'] = '0.1.0'
        f.attrs['format'] = 'hdf5'
        
        def save_recursive(group, data_dict):
            for key, value in data_dict.items():
                if isinstance(value, (np.ndarray, list)):
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], pd.DataFrame):
                            # Handle list of DataFrames
                            list_group = group.create_group(key)
                            for i, df in enumerate(value):
                                df_group = list_group.create_group(f'item_{i}')
                                df_group.create_dataset('data', data=df.values)
                                df_group.create_dataset('columns', 
                                                      data=[col.encode('utf-8') for col in df.columns])
                                df_group.create_dataset('index', data=df.index)
                        else:
                            group.create_dataset(key, data=np.array(value))
                    else:
                        group.create_dataset(key, data=np.array(value))
                elif isinstance(value, pd.DataFrame):
                    df_group = group.create_group(key)
                    df_group.create_dataset('data', data=value.values)
                    df_group.create_dataset('columns', 
                                          data=[col.encode('utf-8') for col in value.columns])
                    df_group.create_dataset('index', data=value.index)
                elif isinstance(value, dict):
                    dict_group = group.create_group(key)
                    save_recursive(dict_group, value)
                elif isinstance(value, (int, float, str, bool)):
                    group.attrs[key] = value
                else:
                    # Try to convert to string representation
                    group.attrs[f'{key}_str'] = str(value)
        
        save_recursive(f, results)


def save_results_pickle(results: Dict[str, Any], filepath: str) -> None:
    """Save results in pickle format."""
    save_dict = results.copy()
    save_dict['_metadata'] = {
        'save_time': datetime.now().isoformat(),
        'version': '0.1.0',
        'format': 'pickle'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_results_matlab(results: Dict[str, Any], filepath: str) -> None:
    """Save results in MATLAB-compatible format."""
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError("scipy.io.savemat required for MATLAB format")
    
    # Convert to MATLAB-compatible format
    matlab_dict = {}
    
    for key, value in results.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], pd.DataFrame):
                # Convert DataFrames to cell array
                matlab_dict[key] = [df.values for df in value]
            else:
                matlab_dict[key] = np.array(value) if not isinstance(value[0], dict) else str(value)
        elif isinstance(value, pd.DataFrame):
            matlab_dict[key] = value.values
            matlab_dict[f'{key}_columns'] = value.columns.tolist()
        elif isinstance(value, dict):
            matlab_dict[f'{key}_dict'] = str(value)
        else:
            matlab_dict[key] = value
    
    # Add metadata
    matlab_dict['metadata_save_time'] = datetime.now().isoformat()
    matlab_dict['metadata_version'] = '0.1.0'
    
    savemat(filepath, matlab_dict)


def load_results(filepath: str, format_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Load optimization results from file.
    
    Parameters
    ----------
    filepath : str
        Input file path
    format_type : str, optional
        Format type (auto-detected if None)
        
    Returns
    -------
    dict
        Results dictionary
    """
    if format_type is None:
        # Auto-detect format from extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.npz':
            format_type = 'npz'
        elif ext in ['.h5', '.hdf5']:
            format_type = 'hdf5'
        elif ext in ['.pkl', '.pickle']:
            format_type = 'pickle'
        elif ext == '.mat':
            format_type = 'matlab'
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")
    
    if format_type == 'npz':
        return load_results_npz(filepath)
    elif format_type == 'hdf5':
        return load_results_hdf5(filepath)
    elif format_type == 'pickle':
        return load_results_pickle(filepath)
    elif format_type == 'matlab':
        return load_results_matlab(filepath)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def load_results_npz(filepath: str) -> Dict[str, Any]:
    """Load results from NumPy compressed format."""
    data = np.load(filepath, allow_pickle=True)
    results = {}
    
    for key in data.files:
        if key.endswith('_list'):
            # Reconstruct list of DataFrames
            base_key = key[:-5]  # Remove '_list' suffix
            results[base_key] = [pd.DataFrame.from_dict(df_dict) 
                               for df_dict in data[key].item()]
        elif key.endswith('_data'):
            # Reconstruct DataFrame
            base_key = key[:-5]  # Remove '_data' suffix
            if f'{base_key}_columns' in data.files:
                results[base_key] = pd.DataFrame(
                    data[key],
                    columns=data[f'{base_key}_columns'].tolist(),
                    index=data[f'{base_key}_index'].tolist() if f'{base_key}_index' in data.files else None
                )
        elif not (key.endswith('_columns') or key.endswith('_index')):
            results[key] = data[key].item() if data[key].ndim == 0 else data[key]
    
    return results


def load_results_hdf5(filepath: str) -> Dict[str, Any]:
    """Load results from HDF5 format."""
    results = {}
    
    with h5py.File(filepath, 'r') as f:
        def load_recursive(group):
            data = {}
            
            # Load attributes
            for key, value in group.attrs.items():
                data[key] = value
            
            # Load datasets and subgroups
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    data[key] = item[...]
                elif isinstance(item, h5py.Group):
                    if 'data' in item and 'columns' in item:
                        # This is a DataFrame
                        df_data = item['data'][...]
                        columns = [col.decode('utf-8') for col in item['columns'][...]]
                        index = item['index'][...] if 'index' in item else None
                        data[key] = pd.DataFrame(df_data, columns=columns, index=index)
                    else:
                        # Regular group or list of DataFrames
                        if all(subkey.startswith('item_') for subkey in item.keys()):
                            # List of DataFrames
                            df_list = []
                            for subkey in sorted(item.keys()):
                                df_group = item[subkey]
                                df_data = df_group['data'][...]
                                columns = [col.decode('utf-8') for col in df_group['columns'][...]]
                                index = df_group['index'][...] if 'index' in df_group else None
                                df_list.append(pd.DataFrame(df_data, columns=columns, index=index))
                            data[key] = df_list
                        else:
                            data[key] = load_recursive(item)
            
            return data
        
        results = load_recursive(f)
    
    return results


def load_results_pickle(filepath: str) -> Dict[str, Any]:
    """Load results from pickle format."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def load_results_matlab(filepath: str) -> Dict[str, Any]:
    """Load results from MATLAB format."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy.io.loadmat required for MATLAB format")
    
    data = loadmat(filepath)
    
    # Remove MATLAB metadata
    results = {k: v for k, v in data.items() 
              if not k.startswith('__')}
    
    return results


def export_summary_report(results: Dict[str, Any], filepath: str) -> None:
    """
    Export human-readable summary report.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str
        Output text file path
    """
    with open(filepath, 'w') as f:
        f.write("PyInverseLLE Optimization Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Metadata
        if '_metadata' in results:
            f.write(f"Generated: {results['_metadata'].get('save_time', 'Unknown')}\n")
            f.write(f"Version: {results['_metadata'].get('version', 'Unknown')}\n\n")
        
        # Optimization parameters
        if 'optimization_params' in results:
            f.write("Optimization Parameters:\n")
            f.write("-" * 25 + "\n")
            for key, value in results['optimization_params'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Results summary
        f.write("Results Summary:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Final Generation: {results.get('final_generation', 'N/A')}\n")
        f.write(f"Best Fitness: {results.get('best_fitness', 'N/A'):.6e}\n")
        
        if 'best_fitness_history' in results:
            history = results['best_fitness_history']
            f.write(f"Initial Fitness: {history[0]:.6e}\n")
            f.write(f"Final Fitness: {history[-1]:.6e}\n")
            f.write(f"Improvement Factor: {history[0] / history[-1]:.2f}x\n")
        
        # Convergence analysis
        if 'fitness_evolution' in results:
            final_gen = results['fitness_evolution'].shape[1] - 1
            final_fitness = results['fitness_evolution'][:, final_gen]
            
            f.write(f"\nFinal Population Statistics:\n")
            f.write(f"Best: {np.min(final_fitness):.6e}\n")
            f.write(f"Worst: {np.max(final_fitness):.6e}\n")
            f.write(f"Mean: {np.mean(final_fitness):.6e}\n")
            f.write(f"Std: {np.std(final_fitness):.6e}\n")
            
            fitness_tol = results['optimization_params']['fitness_tol']
            converged_count = np.sum(final_fitness < fitness_tol)
            f.write(f"Converged individuals: {converged_count}/{len(final_fitness)}\n")
        
        # Best individual parameters
        if 'best_individual_idx' in results:
            best_idx = results['best_individual_idx']
            f.write(f"\nBest Individual (Index {best_idx}):\n")
            f.write("-" * 30 + "\n")
            
            if 'dispersion_params' in results and len(results['dispersion_params']) > 0:
                final_params = results['dispersion_params'][-1]
                best_params = final_params.iloc[best_idx]
                
                for param_name, value in best_params.items():
                    if isinstance(value, np.ndarray) and len(value) > 5:
                        f.write(f"{param_name}: [{value[0]:.4f}, ..., {value[-1]:.4f}] (length {len(value)})\n")
                    else:
                        f.write(f"{param_name}: {value}\n")
    
    print(f"Summary report saved to: {filepath}")


def export_best_parameters_csv(results: Dict[str, Any], filepath: str) -> None:
    """
    Export best individual parameters as CSV.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str
        Output CSV file path
    """
    if 'dispersion_params' not in results or len(results['dispersion_params']) == 0:
        print("No parameter data available for export")
        return
    
    best_idx = results.get('best_individual_idx', 0)
    final_params = results['dispersion_params'][-1]
    best_params = final_params.iloc[best_idx:best_idx+1]  # Keep as DataFrame
    
    best_params.to_csv(filepath, index=False)
    print(f"Best parameters exported to: {filepath}")


def create_backup(filepath: str, backup_dir: str = './backups') -> str:
    """
    Create backup copy of results file.
    
    Parameters
    ----------
    filepath : str
        Original file path
    backup_dir : str
        Backup directory
        
    Returns
    -------
    str
        Backup file path
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    backup_filename = f"{name}_backup_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    import shutil
    shutil.copy2(filepath, backup_path)
    
    print(f"Backup created: {backup_path}")
    return backup_path