import numpy as np
import os
from typing import Union
from multiprocessing import Pool
from functools import partial
from fbm import fbm, FBM


def generate_BM(batch: int, length: int, channels: int, T: float = 1., seed: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    BM_increments = np.random.normal(loc=0, scale=np.sqrt(T/length), size=(batch, length, channels))
    BMs = np.concatenate([np.zeros((batch, 1, channels)), BM_increments.cumsum(axis=1)], axis=1)
    return BMs

def generate_fBm(batch: int, length: int, channels: int, H: float, seed: Union[int, None] = None, T: float = 1., use_multiprocessing: bool = True, method: str = "daviesharte", chunks: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    fBMs = np.zeros((batch, length+1, channels))
    if use_multiprocessing:
        if chunks is None:
            chunks = 1
        assert batch % chunks == 0, f"If selecting multiprocessing chunks please ensure these divide batch, got chunks = {chunks}, batch = {batch}."
        print('Multiprocessing fBm samples...')
        seeds = [seed + i for i in range(chunks)]
        with Pool(processes=os.cpu_count()- 2) as pool:
            # use this instead of simply partial(fbm, n=length, hurst=H, T=T) to control seeds
            results = pool.map(partial(generate_fBm, batch // chunks, length, channels, H, T=T, use_multiprocessing=False), seeds)
        print('...finished multiprocessing fBm samples.')
        for i, fbm_paths in enumerate(results):
            fBMs[i*batch//chunks:(i+1)*(batch//chunks), :, :] = fbm_paths
    else:
        f = FBM(n=length, hurst=H, length=T, method=method)
        for i in range(batch):
            for j in range(channels):
                fBMs[i, :, j] = f.fbm() 
    return fBMs