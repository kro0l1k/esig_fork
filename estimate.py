import os
import sys
import time
import numpy as np
import pickle

from lib.esig import expected_signature_estimate, expected_signature_estimate_variance

if __name__ == "__main__":   
    processes=['BM', 'fBm', 'MCAR', 'Heston']
    Ns = [10, 20, 30, 40, 50, 60]
    max_depth = 3
    dims = 2
    N_max = max(Ns)
    length_max = 2**((N_max)//10 + 1)

    path_sampling_schemes=['iid', 'chop']
    
    save_dir=os.path.join('.', 'simulations', 'data')

    process = processes[int(sys.argv[1])] # select specific job using PBS_ARRAY_INDEX

    start_time=time.time()

    # generate most granular sample
    for path_sampling_scheme in path_sampling_schemes:        
        if path_sampling_scheme == 'iid':
            paths=np.load(os.path.join(save_dir, f'{process}_paths.npy'), mmap_mode='r')
        elif path_sampling_scheme == 'chop':
            paths=np.load(os.path.join(save_dir, f'{process}_paths_long.npy'), mmap_mode='r')
        else:
            raise ValueError(f'Unkown path_sampling_scheme={path_sampling_scheme}.')

        print(f'Loading {process} {path_sampling_scheme} data took {time.time() - start_time}.')
        
        esig_estimates = {}
        esig_var_estimates = {}
        esig_martingale_estimates = {}
        
        for N in Ns:
            length_N = 2**(N//10 + 1)
            if path_sampling_scheme == 'iid':
                paths_N = paths[:, :N, ::length_max//length_N, :]
                chop = None
            elif path_sampling_scheme == 'chop':
                paths_N = paths[:, :(N*length_max + 1):(length_max//length_N), :]
                chop = length_N
            else:
                raise ValueError(f'Unkown path_sampling_scheme={path_sampling_scheme}.')
            start_time = time.time()
            esig_estimates[f'N={N}'] = expected_signature_estimate(paths_N, max_depth, martingale_indices=None, chop=chop)
            esig_var_estimates[f'N={N}'] = expected_signature_estimate_variance(paths_N, max_depth, chop=chop)
            esig_martingale_estimates[f'N={N}'] = expected_signature_estimate(paths_N, max_depth, martingale_indices=list(range(dims)), chop=chop)
            print(f'Estimating esigs with N = {N} took {time.time() - start_time}.')
            
        np.savez(os.path.join(save_dir, f'{process}_esig_estimates_{path_sampling_scheme}.npz'), **esig_estimates)
        np.savez(os.path.join(save_dir, f'{process}_esig_var_estimates_{path_sampling_scheme}.npz'), **esig_var_estimates)
        np.savez(os.path.join(save_dir, f'{process}_esig_martingale_estimates_{path_sampling_scheme}.npz'), **esig_martingale_estimates)