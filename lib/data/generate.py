from typing import Union, List, Tuple
import warnings
import numpy as np
import scipy
import scipy.linalg
import scipy.stats
import QuantLib as ql


def generate_BM(batch: int, length: int, dims: int, T: float = 1., seed: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    BM_increments = np.random.normal(loc=0, scale=np.sqrt(T/length), size=(batch, length, dims))
    BMs = np.concatenate([np.zeros((batch, 1, dims)), BM_increments.cumsum(axis=1)], axis=1)
    return BMs


def generate_fBm(batch: int, length: int, dims: int, H: float, seed: Union[int, None] = None, T: float = 1.) -> np.ndarray:
    np.random.seed(seed)
    
    scale = (T / length) ** H
    gn = np.random.normal(0.0, 1.0, (batch, length, dims))

    if H == 0.5:
        fBM_increments = gn * scale
    else:
        G = np.zeros((length, length))
        indices = np.arange(length)
        i, j = np.meshgrid(indices, indices, indexing='ij')
        G = 0.5 * (np.abs(i - j - 1)**(2*H) - 2 * np.abs(i - j)**(2*H) + np.abs(i - j + 1)**(2*H))

        C = np.linalg.cholesky(G)
        gn = np.einsum('ij,njd->nid', C, gn)
        fBM_increments = gn * scale
    
    fBms = np.concatenate([np.zeros((batch, 1, dims)), fBM_increments.cumsum(axis=1)], axis=1)
    return fBms


def generate_MCAR(batch: int, length: int, dims: int, AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]], a: Union[np.ndarray, None] = None, Sigma: Union[np.ndarray, None] = None, seed: Union[int, None] = None, T: float = 1., x0: Union[np.ndarray, None] = None) -> np.ndarray:
    np.random.seed(seed)
    P = np.linspace(0, T, length + 1)
    if a is None:
        a = np.zeros(dims)
    if Sigma is None:
        Sigma = np.eye(dims) 
    return simulate_MCAR(batch=batch, P=P, AA=AA, a=a, Sigma=Sigma, x0=x0, uniform=True)


def generate_Heston(batch: int, length: int, dims: int, theta: float = 0.1, kappa: float = 0.6, sigma: float = 0.2, rho: float = -0.15, S0: float = 1., v0: float = 0.1, seed: Union[int, None] = None, T: float = 1.):
    # either return only the price process or the price process and the variance process
    assert dims in [1, 2]
    
    # QuantLib implements the Heston process under the risk neutral measure Q (for pricing)
    # 1. setting dividend_yield = 0.0 this means the *discounted* price process is a martingale;
    # 2. setting also risk_free_rate = 0.0 this implies the price process is a martingale. 
    risk_free_rate = 0.0
    dividend_yield = 0.0
    day_count = ql.Actual360()
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), risk_free_rate, day_count)
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), dividend_yield, day_count)
    )
    
    heston_process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho,
    )

    # use the ql.HestonProcess.QuadraticExponentialMartingale discretization by Andersen
    # note seed=0 in Quantlib means no seed...
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(2*length, ql.UniformRandomGenerator(seed=0 if seed is None else seed + 1))
    )
    time_grid = ql.TimeGrid(T, length)
    generator = ql.GaussianMultiPathGenerator(heston_process, time_grid, rng)

    # Generate paths
    all_paths = []
    for _ in range(batch):
        sample_path = generator.next()
        price_path = np.array(sample_path.value()[0])
        vol_path = np.array(sample_path.value()[1])
        if dims == 1:
            all_paths.append(price_path[:, None])
        elif dims == 2:
            all_paths.append(np.stack([price_path, vol_path], axis=1))

    return np.stack(all_paths)


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