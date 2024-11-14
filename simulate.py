import time
import pickle
from functools import partial
import numpy as np

from lib.esig import expected_signature_estimate 
from lib.data.generate import generate_BM, generate_fBm, generate_MCAR
from lib.data.utils import chop_and_shift

if __name__ == "__main__":
    processes = ['BM', 'MCAR', 'fBm']
    Ns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    n_samples=10_000
    max_depth=5
    dims=2
    seed=0
    H=0.75
    AA=(np.ones(dims), np.ones(dims))
    path_sampling_schemes=['iid', 'chop']
    estimators=['classic', 'martingale']

    for process in processes:
        start_time=time.time()

        # generate most granular sample
        N_max=max(Ns)
        if process == 'BM':
            generate_fn=generate_BM
        elif process == 'fBm':
            generate_fn=partial(generate_fBm, H=H, use_multiprocessing=True, chunks=N_max*10)
        elif process == 'MCAR':
            generate_fn=partial(generate_MCAR, AA=AA)
        else:
            raise ValueError(f'Unkown process={process}.')
        for path_sampling_scheme in path_sampling_schemes:        
            if path_sampling_scheme == 'iid':
                paths=generate_fn(batch=n_samples*N_max, length=2**N_max, dims=dims, T=1, seed=seed).reshape((n_samples, N_max, 2**N_max + 1, dims))
                with open(f'./simulations/data/{process}_paths.pickle', 'wb') as f:
                    pickle.dump(paths, f)
            elif path_sampling_scheme == 'chop':
                paths=generate_fn(batch=n_samples, length=N_max * 2**N_max, dims=dims, T=N_max, seed=seed)
                with open(f'./simulations/data/{process}_paths_long.pickle', 'wb') as f:
                    pickle.dump(paths, f)
                paths=chop_and_shift(paths, chops=2**N_max)
            else:
                raise ValueError(f'Unkown path_sampling_scheme={path_sampling_scheme}.')

            print(f'Generating data took {time.time() - start_time}.')

        # start_time=time.time()
        
        # for estimator in estimators:
        #     esig_estimates={}
        #     for N in Ns:
        #         paths_N=paths[:, :N, ::2**(N_max - N), :]
        #         martingale_indices = list(range(dims)) if estimator == 'martingale' else None
        #         esig_estimates[N]=expected_signature_estimate(paths_N, max_depth, martingale_indices=None)
            
        #     with open(f'./simulations/data/{process}_esig_estimates_{estimator}_{path_sampling}.pickle', 'wb') as f:
        #         pickle.dump(esig_estimates, f)

        # print(f'Estimating signatures took {time.time() - start_time}.')