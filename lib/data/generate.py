import os
from typing import Union, List, Tuple
from multiprocessing import Pool
from functools import partial
import numpy as np
import scipy
import scipy.linalg
import scipy.stats
from fbm import fbm, FBM


def generate_BM(batch: int, length: int, dims: int, T: float = 1., seed: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    BM_increments = np.random.normal(loc=0, scale=np.sqrt(T/length), size=(batch, length, dims))
    BMs = np.concatenate([np.zeros((batch, 1, dims)), BM_increments.cumsum(axis=1)], axis=1)
    return BMs


def generate_fBm(batch: int, length: int, dims: int, H: float, seed: Union[int, None] = None, T: float = 1., use_multiprocessing: bool = True, method: str = "daviesharte", chunks: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    fBMs = np.zeros((batch, length+1, dims))
    if use_multiprocessing:
        if chunks is None:
            chunks = 1
        assert batch % chunks == 0, f"If selecting multiprocessing chunks please ensure these divide batch, got chunks = {chunks}, batch = {batch}."
        print('Multiprocessing fBm samples...')
        seeds = [seed + i for i in range(chunks)]
        with Pool(processes=os.cpu_count() - 2) as pool:
            # use this instead of simply partial(fbm, n=length, hurst=H, T=T) to control seeds
            results = pool.map(partial(generate_fBm, batch // chunks, length, dims, H, T=T, use_multiprocessing=False), seeds)
        print('...finished multiprocessing fBm samples.')
        for i, fbm_paths in enumerate(results):
            fBMs[i*batch//chunks:(i+1)*(batch//chunks), :, :] = fbm_paths
    else:
        f = FBM(n=length, hurst=H, length=T, method=method)
        for i in range(batch):
            for j in range(dims):
                fBMs[i, :, j] = f.fbm() 
    return fBMs


def generate_MCAR(batch: int, length: int, dims: int, AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]], a: Union[np.ndarray, None] = None, Sigma: Union[np.ndarray, None] = None, seed: Union[int, None] = None, T: float = 1., x0: Union[np.ndarray, None] = None) -> np.ndarray:
    np.random.seed(seed)
    P = np.linspace(0, T, length + 1)
    if a is None:
        a = np.zeros(dims)
    if Sigma is None:
        Sigma = np.eye(dims) 
    return simulate_MCAR(batch=batch, P=P, AA=AA, a=a, Sigma=Sigma, x0=x0, uniform=True)


# Compute MCAR structural matrix from parameters AA
def MCAR_A(AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]]) -> np.ndarray:
    """
    Construct MCAR structural matrix from drift parameters A.
    :param AA: MCAR drift parameters, list or tuple of p (d, d) np.ndarray
    :return A_AA: MCAR structural matrix, (pd, pd) np.ndarray
    """
    # get dimensions
    p = len(AA)
    d = 1 if isinstance(AA[0], float) else AA[0].shape[0]
    # compute MCAR structural matrix
    A = np.zeros([p*d, p*d])
    for i in range(d, p*d, d):
        A[(i-d):i, i:(i+d)] = np.eye(d)
    for i in range(p):
        A[-d:, i*d:(i+1)*d] = - AA[p-i-1]
    return A


def simulate_MCAR_stat_distr(batch: int, AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]], a: np.ndarray, Sigma: np.ndarray):
    """
    Simulate from stationary distribution of MCAR with Brownian driver process exactly.
    :param batch: number of samples to generate, int
    :param AA: MCAR drift parameters, list or tuple of p (d, d) np.ndarray
    :param a: drift of Brownian driver, (d,) np.ndarray
    :param Sigma: covariance of Brownian driver, (d, d) np.ndarray
    :return x: sample from stationary distribution, (batch, d) np.ndarray
    """
    if isinstance(a, float):
        a = np.array([a])
    p = len(AA)
    d = len(a)
    pd = p*d
    A = MCAR_A(AA)
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)
    T = - np.log(1e-8) / np.linalg.norm(A)

    M = np.zeros([2*pd, 2*pd])
    M[:pd, :pd] = A
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = - A.T
    V = scipy.linalg.expm(M*T)[:pd, pd:].dot(scipy.linalg.expm(A*T).T)

    a_component = - np.linalg.inv(A).dot(E).dot(a)
    W_component = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=batch)

    x = np.expand_dims(a_component, axis=0) + W_component
    return x


def simulate_MCAR(batch: int, P: np.ndarray, AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]], a: np.ndarray, Sigma: np.ndarray, x0: Union[np.ndarray, None] = None, uniform: bool = False) -> np.ndarray:
    """
    Simulate (discrete) paths from a MCAR model with structural matrix A and driving Brownian motion with drift a and covariance Sigma.
    :param batch: number of samples to generate, int
    :param P: partition over which to simulate the MCAR process [0 = t_0, t_1, ..., t_length = T], (length+1,) np.ndarray
    :param A: MCAR drift parameters, list or tuple of p (d, d) np.ndarray
    :param x0: state space initial condition - if None simulate from stationary distribution, p x (d,) np.ndarray
    :param a: drift of Brownian driver, (d,) np.ndarray
    :param Sigma: covariance of Brownian driver, (d, d) np.ndarray
    :return Y: MCAR simulation, (d, length+1) np.ndarray
    """
    # for 1-dimensional allow parameters to be floats
    if isinstance(a, float):
        a = np.array([a])
    if isinstance(Sigma, float):
        Sigma = np.array([[Sigma]])

    # get dimensions and time step
    d = len(a)
    p = len(AA)
    pd = p*d
    length = len(P) - 1

    if x0 is None:
        x0 = simulate_MCAR_stat_distr(batch=batch, AA=AA, a=a, Sigma=Sigma)
    
    # compute structural matrix
    A = MCAR_A(AA)
    
    # compute E, Sigma_tilde
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)
    
    # initialize 
    X = np.zeros([batch, length + 1, pd])
    X[:, 0, :] = x0

    # precompute for speed-up
    eA = scipy.linalg.expm(A)
    A_inv = np.linalg.inv(A)
    M = np.zeros([2*pd, 2*pd])
    M[:pd, :pd] = A
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = - A.T
    eM = scipy.linalg.expm(M)
    
    if uniform:
        delta_t = P[1] - P[0]
        eAt = scipy.linalg.fractional_matrix_power(eA, delta_t).real
        
        # increment due to b
        a_increment = (eAt - np.eye(pd)).dot(A_inv).dot(E).dot(a)
        
        # increment due to W
        V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eAt.T) + 1e-12*np.eye(pd)
        W_increments = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=batch*length).reshape(batch, length, pd)
        
        # evolve the process
        for n in range(length):
            X[:, n+1, :] = np.einsum('ij,kj->ki', eAt, X[:, n, :]) + np.expand_dims(a_increment, axis=0) + W_increments[:, n, :]
    else: 
        for n in range(length):
            delta_t = P[n+1] - P[n]
            eAt = scipy.linalg.fractional_matrix_power(eA, delta_t).real

            # increment due to b
            a_increment = (eAt - np.eye(pd)).dot(A_inv).dot(E).dot(a)
            
            # increment due to W
            M = np.zeros([2*pd, 2*pd])
            M[:pd, :pd] = A
            M[:pd, pd:] = Sigma_tilde
            M[pd:, pd:] = - A.T
            V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eAt.T) + 1e-12*np.eye(pd)
            W_increment = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

            # evolve the process
            X[:, n+1] = eAt.dot(X[:, n]) + a_increment + W_increment

    return X[:, :, :d]