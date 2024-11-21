from typing import Union, List, Tuple
import numpy as np
import QuantLib as ql
from utils import simulate_MCAR

def generate_BM(batch: int, length: int, dims: int, T: float, seed: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    BM_increments = np.random.normal(loc=0, scale=np.sqrt(T/length), size=(batch, length, dims))
    BMs = np.concatenate([np.zeros((batch, 1, dims)), BM_increments.cumsum(axis=1)], axis=1)
    return BMs


def generate_fBm(batch: int, length: int, dims: int, T: float, H: float, seed: Union[int, None] = None) -> np.ndarray:
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


def generate_MCAR(batch: int, length: int, dims: int,  T: float, AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]], x0: Union[np.ndarray, None] = None, a: Union[np.ndarray, None] = None, Sigma: Union[np.ndarray, None] = None, seed: Union[int, None] = None) -> np.ndarray:
    np.random.seed(seed)
    P = np.linspace(0, T, length + 1)
    if a is None:
        a = np.zeros(dims)
    if Sigma is None:
        Sigma = np.eye(dims) 
    return simulate_MCAR(batch=batch, P=P, AA=AA, a=a, Sigma=Sigma, x0=x0, uniform=True)


def generate_Heston(batch: int, length: int, dims: int, T: float, theta: float, kappa: float, sigma: float, rho: float, S0: float, v0: float, seed: Union[int, None] = None):
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