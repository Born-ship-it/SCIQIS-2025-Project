import numpy as np
import pickle
import hashlib
import os
from pathlib import Path

from Fock_space_func import result_qed


def get_parameter_hash(*args, **kwargs):
    """Generate a unique hash for the function parameters."""
    param_str = f"{args}_{sorted(kwargs.items())}".encode()
    return hashlib.md5(param_str).hexdigest()

def cached_result_qed(N_vals, M_vals, Ns, Ms, cache_dir="cache_of_results", force_recompute=False, **kwargs):
    """
    Wrapper function that caches results to avoid recomputation.
    
    Parameters:
        cache_dir: Directory to store cached results
        force_recompute: If True, ignore cache and recompute
        **kwargs: Additional parameters passed to result_qed
    """
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(exist_ok=True)
    
    # Generate unique hash for these parameters
    param_hash = get_parameter_hash(N_vals.tobytes(), M_vals.tobytes(), 
                                  Ns.tobytes(), Ms.tobytes(), **kwargs)
    cache_file = Path(cache_dir) / f"result_qed_{param_hash}.pkl"
    metadata_file = Path(cache_dir) / f"result_qed_{param_hash}_meta.pkl"
    
    # Check if cached result exists and should be used
    if not force_recompute and cache_file.exists() and metadata_file.exists():
        try:
            # Load metadata to check if parameters match
            with open(metadata_file, 'rb') as f:
                cached_metadata = pickle.load(f)
            
            current_metadata = {
                'N_vals': N_vals,
                'M_vals': M_vals,
                'Ns_shape': Ns.shape,
                'Ms_shape': Ms.shape,
                'kwargs': kwargs
            }
            
            # Check if parameters match
            params_match = (
                np.array_equal(cached_metadata['N_vals'], current_metadata['N_vals']) and
                np.array_equal(cached_metadata['M_vals'], current_metadata['M_vals']) and
                cached_metadata['Ns_shape'] == current_metadata['Ns_shape'] and
                cached_metadata['Ms_shape'] == current_metadata['Ms_shape'] and
                cached_metadata['kwargs'] == current_metadata['kwargs']
            )
            
            if params_match:
                print(f"Loading cached results from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
               
        except (pickle.PickleError, EOFError, KeyError) as e:
            print(f"Cache corrupted: {e}. Recomputing...")
    
    # Compute results (this takes ~30 mins)
    print("Computing results (this may take ~30 minutes)...")
    results = result_qed(N_vals, M_vals, Ns, Ms, **kwargs)
    
    # Save results and metadata
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        metadata = {
            'N_vals': N_vals,
            'M_vals': M_vals,
            'Ns_shape': Ns.shape,
            'Ms_shape': Ms.shape,
            'kwargs': kwargs
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Results saved to {cache_file}")
        
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    return results

# Alternative version using decorator pattern
def cache_results(func):
    """Decorator to add caching to any function."""
    def wrapper(N_vals, M_vals, Ns, Ms, cache_dir="cache", force_recompute=False, **kwargs):
        Path(cache_dir).mkdir(exist_ok=True)
        
        param_hash = get_parameter_hash(N_vals.tobytes(), M_vals.tobytes(), 
                                      Ns.tobytes(), Ms.tobytes(), **kwargs)
        cache_file = Path(cache_dir) / f"{func.__name__}_{param_hash}.pkl"
        
        if not force_recompute and cache_file.exists():
            try:
                print(f"Loading cached results from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError) as e:
                print(f"Cache corrupted: {e}. Recomputing...")
        
        print(f"Computing {func.__name__} (this may take a while)...")
        results = func(N_vals, M_vals, Ns, Ms, **kwargs)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Results saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
        
        return results
    return wrapper