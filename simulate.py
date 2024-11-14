import time
import pickle
from functools import partial

from lib.esig import expected_signature_estimate 
from lib.data.generate import generate_BM, generate_fBm
from lib.data.utils import chop_and_shift

if __name__ == "__main__":
    process='fBm' # 'BM', 'fBm'
    Ns=[2, 4]
    n_samples=10_000
    max_depth=3
    dims=2
    seed=0
    H=0.75
    independent_paths=True


    start_time=time.time()

    # generate most granular sample
    N_max=max(Ns)
    if process == 'BM':
        generate_fn=generate_BM
    elif process == 'fBm':
        generate_fn=partial(generate_fBm, H=H, use_multiprocessing=True, chunks=10)
    else:
        raise ValueError(f'Unkown process={process}.')
    
    if independent_paths:
        paths=generate_fn(batch=n_samples*N_max, length=2**N_max, channels=dims, T=1, seed=seed).reshape((n_samples, N_max, 2**N_max + 1, dims))
    else:
        paths=generate_fn(batch=n_samples, length=N_max * 2**N_max, channels=dims, T=N_max, seed=seed)
        paths=chop_and_shift(paths, chops=2**N_max)

    print(f'Generating data took {time.time() - start_time}.')

    with open(f'./simulations/data/{process}_paths.pickle', 'wb') as f:
        pickle.dump(paths, f)

    start_time=time.time()

    esig_estimates={}
    martingale_esig_estimates={}

    for N in Ns:
        paths_N=paths[:, :N, ::2**(N_max - N), :]
        esig_estimates[N]=expected_signature_estimate(paths_N, max_depth, martingale_indices=None)
        martingale_esig_estimates[N]=expected_signature_estimate(paths_N, max_depth, martingale_indices=list(range(dims)))

    print(f'Estimating signatures took {time.time() - start_time}.')

    with open(f'./simulations/data/{process}_esig_estimates.pickle', 'wb') as f:
        pickle.dump(esig_estimates, f)

    with open(f'./simulations/data/{process}_martingale_esig_estimates.pickle', 'wb') as f:
        pickle.dump(martingale_esig_estimates, f)