import os
import time
from functools import partial
import numpy as np

from lib.data.generate import generate_BM, generate_fBm, generate_MCAR, generate_Heston

if __name__ == "__main__":   
    processes=['BM', 'fBm', 'MCAR', 'Heston']
    n_samples=10_000
    max_depth=5
    N_max=60
    length_max=2**((N_max)//10 + 1)
    T=1
    dims=2
    seed=0
    
    # fBm
    H=0.75
    
    # MCAR
    AA=(np.array([[1., 0.], [1., 1.]]), np.array([[1., 1.], [0., 1.]]))
    
    # Heston
    theta = 0.1
    kappa = 0.6
    sigma = 0.2
    rho = -0.15
    S0 = 1.
    v0 = 0.1 

    path_sampling_schemes=['iid', 'chop']
    estimators=['classic', 'martingale']
    
    save_dir=os.path.join('.', 'simulations', 'data')
    os.makedirs(save_dir, exist_ok=True)

    for process in processes:
        start_time=time.time()

        # generate most granular sample
        if process == 'BM':
            generate_fn=generate_BM
        elif process == 'fBm':
            generate_fn=partial(generate_fBm, H=H)
        elif process == 'MCAR':
            generate_fn=partial(generate_MCAR, AA=AA)
        elif process == 'Heston':
            generate_fn=partial(generate_Heston, theta=theta, kappa=kappa, sigma=sigma, rho=rho, S0=S0, v0=v0)
        else:
            raise ValueError(f'Unkown process={process}.')
        for path_sampling_scheme in path_sampling_schemes:        
            if path_sampling_scheme == 'iid':
                paths=generate_fn(batch=n_samples*N_max, length=length_max, dims=dims, T=T, seed=seed).reshape((n_samples, N_max, length_max + 1, dims))
                np.save(os.path.join(save_dir, f'{process}_paths.npy'), paths)
            elif path_sampling_scheme == 'chop':
                paths=generate_fn(batch=n_samples, length=N_max*length_max, dims=dims, T=T*N_max, seed=seed)
                np.save(os.path.join(save_dir, f'{process}_paths_long.npy'), paths)
            else:
                raise ValueError(f'Unkown path_sampling_scheme={path_sampling_scheme}.')

            print(f'Generating {process} {path_sampling_scheme} data took {time.time() - start_time}.')

        # start_time=time.time()

        # Ns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        # for estimator in estimators:
        #     esig_estimates={}
        #     for N in Ns:
        #         paths_N=paths[:, :N, ::2**(N_max - N), :]
        #         martingale_indices = list(range(dims)) if estimator == 'martingale' else None
        #         esig_estimates[N]=expected_signature_estimate(paths_N, max_depth, martingale_indices=None)
            
        #     with open(f'./simulations/data/{process}_esig_estimates_{estimator}_{path_sampling}.pickle', 'wb') as f:
        #         pickle.dump(esig_estimates, f)

        # print(f'Estimating signatures took {time.time() - start_time}.')