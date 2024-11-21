import signatory
import iisignature
import torch
import numpy as np
from typing import overload, List, Union
import warnings

from lib.utils import sig_idx_to_word, sig_word_to_idx, shuffle, chop_and_shift, chop_shift_and_augment

def get_signature_indices(depth: int, channels: int, ending_indices: List[int]) -> List[int]:
    if not isinstance(ending_indices, list) or not all(isinstance(i, int) for i in ending_indices) or not all(0 <= i < channels for i in ending_indices):
        raise ValueError(f'The ending_indices argument must be a list of integers between 0 and channels={channels}.')
    sig_indices = []
    start_index = 0
    for i in range(1, depth + 1):
        # in each signature level, of size channels**i, we want to extract j*channels**(i-1):(j+1)*channels**(i-1) for j in ending_indices
        for j in ending_indices:
            sig_indices.extend(list(range(start_index+j*channels**(i-1), start_index+(j+1)*channels**(i-1))))
        start_index += channels**i
    return sig_indices


def _expected_signature_estimate_torch(path: torch.Tensor, depth: int, martingale_indices: List[int] = None, stream: bool = False, return_samples: bool = False) -> torch.Tensor:    
    length, dim = path.shape[-2], path.shape[-1]                                                                                            # shape: (..., samples, length, d)
    signatures_stream = signatory.signature(path.reshape(-1, length, dim), depth, stream=True)                                              # shape: (..., samples, length - 1, d + ... + d**M)
    signatures_stream = signatures_stream.reshape((*path.shape[:-2], *signatures_stream.shape[-2:]))                                        # shape: (..., samples, length - 1, d + ... + d**M)
    signatures = signatures_stream if stream else signatures_stream[..., -1, :]                                                             # shape: (..., samples, length - 1, d + ... + d**M) or (..., samples, d + ... + d**M) 
    sample_dim = -3 if stream else -2
    if martingale_indices:
        signatures_lower = signatures_stream[..., :-1, :-dim**depth]                                                                        # shape: (..., samples, length - 2, d + ... + d**(M-1))
        # pre-pend signature starting values at zero
        signatures_start = torch.cat([torch.zeros(*path.shape[:-2], 1, dim**i).to(path.device) for i in range(1, depth)], dim=-1)           # shape: (..., samples, 1, d + ... + d**(M-1))	
        signatures_lower = torch.cat([signatures_start, signatures_lower], dim=-2)                                                          # shape: (..., samples, length - 1, d + ... + d**(M-1))        
        signatures_lower = torch.cat([torch.ones(*path.shape[:-2], length - 1, 1).to(path.device), signatures_lower], dim=-1)               # shape: (..., samples, length - 1, 1 + d + ... + d**(M-1))
        corrections = torch.einsum('...k,...l->...kl', signatures_lower, torch.diff(path, dim=-2))                                          # shape: (..., samples, length - 1, 1 + d + ... + d**(M-1), d)
        if stream:
            corrections = torch.cumsum(corrections, dim=-3).reshape((*path.shape[:-2], length - 1, -1))                                     # shape: (..., samples, length - 1, d + ... + d**M)
        else:
            corrections = torch.sum(corrections, dim=-3).reshape((*path.shape[:-2], -1))                                                    # shape: (..., samples, d + ... + d**M)
        num = torch.einsum('...,...->...', signatures, corrections).mean(dim=sample_dim)                                                    # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)
        denom = (corrections**2).mean(axis=sample_dim)                                                                                      # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)
        #NOTE: replacing zeros in denom with ones both c_hat and corrections are zero at those indices
        denom[denom == 0] = 1
        c_hat = (num / denom).unsqueeze(dim=sample_dim)                                                                                     # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)
        sig_indices = get_signature_indices(depth, dim, martingale_indices)
        signatures[..., sig_indices] -= (c_hat * corrections)[..., sig_indices]                                                             # shape: (..., samples, length - 1, d + ... + d**M) or (..., samples, d + ... + d**M)
    else:
        pass
    if return_samples:
        return signatures                                                                                                                   # shape: (..., samples, length - 1, d + ... + d**M) or (..., samples, d + ... + d**M)
    else:
        return signatures.mean(dim=sample_dim)                                                                                              # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)


def _expected_signature_estimate_numpy(path: np.ndarray, depth: int, martingale_indices: List[int] = None, stream: bool = False, return_samples: bool = False) -> np.ndarray:    
    length, dim = path.shape[-2], path.shape[-1]                                                                                            # shape: (..., samples, length, d)
    signatures_stream = iisignature.sig(path.reshape(-1, length, dim), depth, 2)                                                            # shape: (..., samples, length - 1, d + ... + d**M)
    signatures_stream = signatures_stream.reshape((*path.shape[:-2], *signatures_stream.shape[-2:]))                                        # shape: (..., samples, length - 1, d + ... + d**M)
    signatures = signatures_stream if stream else signatures_stream[..., -1, :]                                                             # shape: (..., samples, length - 1, d + ... + d**M) or (..., samples, d + ... + d**M) 
    sample_axis = -3 if stream else -2
    if martingale_indices:
        signatures_lower = signatures_stream[..., :-1, :-dim**depth]                                                                        # shape: (..., samples, length - 2, d + ... + d**(M-1))
        # pre-pend signature starting values at zero
        signatures_start = np.concatenate([np.zeros((*path.shape[:-2], 1, dim**i)) for i in range(1, depth)], axis=-1)                      # shape: (..., samples, 1, d + ... + d**(M-1))	
        signatures_lower = np.concatenate([signatures_start, signatures_lower], axis=-2)                                                    # shape: (..., samples, length - 1, d + ... + d**(M-1))        
        signatures_lower = np.concatenate([np.ones((*path.shape[:-2], length - 1, 1)), signatures_lower], axis=-1)                          # shape: (..., samples, length - 1, 1 + d + ... + d**(M-1))
        corrections = np.einsum('...k,...l->...kl', signatures_lower, np.diff(path, axis=-2))                                               # shape: (..., samples, length - 1, 1 + d + ... + d**(M-1), d)
        if stream:
            corrections = np.cumsum(corrections, axis=-3).reshape((*path.shape[:-2], length - 1, -1))                                       # shape: (..., samples, length - 1, d + ... + d**M)
        else:
            corrections = np.sum(corrections, axis=-3).reshape((*path.shape[:-2], -1))                                                      # shape: (..., samples, d + ... + d**M)
        num = np.einsum('...,...->...', signatures, corrections).mean(axis=sample_axis)                                                     # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)
        denom = (corrections**2).mean(axis=sample_axis)                                                                                     # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)
        #NOTE: replacing zeros in denom with ones both c_hat and corrections are zero at those indices
        denom[denom == 0] = 1
        c_hat = np.expand_dims(num / denom, axis=sample_axis)                                                                               # shape: (..., 1, length - 1, d + ... + d**M) or (..., 1, d + ... + d**M)
        sig_indices = get_signature_indices(depth, dim, martingale_indices)
        signatures[..., sig_indices] -= (c_hat * corrections)[..., sig_indices]                                                             # shape: (..., samples, length - 1, d + ... + d**M) or (..., samples, d + ... + d**M)
    else:
        pass
    if return_samples:
        return signatures                                                                                                                   # shape: (..., samples, length - 1, d + ... + d**M) or (..., samples, d + ... + d**M)
    else:
        return signatures.mean(axis=sample_axis)                                                                                            # shape: (..., length - 1, d + ... + d**M) or (..., d + ... + d**M)


@overload
def expected_signature_estimate(path: torch.Tensor, depth: int, martingale_indices: List[int] = None, stream: bool = False, return_samples: bool = False) -> torch.Tensor: ...

@overload
def expected_signature_estimate(path: np.ndarray, depth: int, martingale_indices: List[int] = None, stream = False, return_samples: bool = False) -> np.ndarray: ...

def expected_signature_estimate(path: Union[np.ndarray, torch.Tensor], depth: int, martingale_indices: List[int] = None, stream: bool = False, chop: Union[int, None] = None, return_samples: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    :param path: collection of paths over which to compute the expected signature, shape (..., samples, length, d) if chop = None else (..., length, d)
    :param depth: depth at which to truncate the expected signature
    :param martingale_indices: to which dimensions of the path we wish to apply martingale correction
    :param stream: whether to return the stream of expected signatures or just the expected signature over the full path
    :param chop: if chop is not None it must be an int dividing length - 1 representing the points at which to chop and shift the path to form samples of length chop, the path is reshaped as (..., samples = (length - 1) // chop, chop + 1, d)
    :return esigs: the expected signatures computed by averaging over samples, shape (..., length - 1, d + ... + d**depth) if stream = True else (..., d + ... + d**depth)
    """
    if chop is not None:
        path = chop_and_shift(path, chops=chop)
    if isinstance(path, torch.Tensor):
        return _expected_signature_estimate_torch(path=path, depth=depth, martingale_indices=martingale_indices, stream=stream, return_samples=return_samples)
    elif isinstance(path, np.ndarray):
        return _expected_signature_estimate_numpy(path=path, depth=depth, martingale_indices=martingale_indices, stream=stream, return_samples=return_samples)
    else:
        raise ValueError(f'Only torch.Tensor and np.ndarray types supported for path, got = {type(path)}.')
    
import time

def expected_signature_estimate_variance(path: np.ndarray, depth: int, martingale_indices: List[int] = None, stream: bool = False, chop: Union[int, None] = None, sample: bool = True, overlapping_samples: bool = False) -> np.ndarray:
    #NOTE: sample=True is faster and yields the same result as sample=False with overlapping_samples=False.
    dim = path.shape[-1]
    sig_dim = sum([dim**i for i in range(1, depth+1)])
    if chop is None:
        if sample:
            sigs = expected_signature_estimate(path=path, depth=depth, martingale_indices=martingale_indices, stream=stream, return_samples=True)
            sigs = np.swapaxes(sigs, -3, -2) if stream else sigs
            esigs = sigs.mean(-2, keepdims=True)
            cov = np.einsum('...i,...j->...ij', sigs - esigs, sigs - esigs).mean(-3)
        else:
            esig = expected_signature_estimate(path=path, depth=2*depth, martingale_indices=martingale_indices, stream=stream)
            cov = np.zeros((*esig.shape[:-1], sig_dim, sig_dim))
            for i in range(sig_dim):
                I = sig_idx_to_word(i, dim)
                for j in range(i, sig_dim):
                    J = sig_idx_to_word(j, dim)
                    K_set = shuffle(I, J)
                    k_set = [sig_word_to_idx(K, dim) for K in K_set]
                    cov[:, i, j] = cov[:, j, i] = esig[..., k_set].sum(axis=-1) - esig[..., i] * esig[..., j]
    else:
        length = path.shape[-2]
        N = (length - 1) // chop
        n_max = int(N**0.25)
        if sample:
            sigs = expected_signature_estimate(path=path, depth=depth, martingale_indices=martingale_indices, stream=stream, chop=chop, return_samples=True)
            sigs = np.swapaxes(sigs, -3, -2) if stream else sigs
            esigs = sigs.mean(-2, keepdims=True)
            cov = np.einsum('...i,...j->...ij', sigs - esigs, sigs - esigs).mean(-3)
            #NOTE: use Newey-West / Bartlett kernel to ensure positive semi-definiteness when using overlapping samples
            for n in range(1, n_max):
                if overlapping_samples:
                    cov_n = np.einsum('...i,...j->...ij', sigs[..., :(N-n), :] - esigs, sigs[..., n:, :] - esigs).mean(-3)
                else:
                    cov_n = np.einsum('...i,...j->...ij', sigs[..., :(N-n):(n+1), :] - esigs, sigs[..., n::(n+1), :] - esigs).mean(-3)
                cov += (1 - n/n_max) * (cov_n + np.swapaxes(cov_n, -2, -1))
        else:
            esigs = {}
            for n in range(n_max):
                try:
                    esigs[n] = expected_signature_estimate(path=chop_shift_and_augment(path, n, chop), depth=2*depth, martingale_indices=martingale_indices, stream=stream)
                except MemoryError:
                    warnings.warn(f'Cutting off long-run covariance at level n = {n-1} due to insufficient RAM.')
                    n_max = n
                    break
            covs = {n: np.zeros((*esigs[0].shape[:-1], sig_dim, sig_dim)) for n in range(N)}
            for i in range(sig_dim):
                I = sig_idx_to_word(i, dim)
                for j in range(sig_dim):
                    J = sig_idx_to_word(j, dim)
                    for n in range(N):
                        J_nd = tuple(j + n*dim for j in J)
                        K_set = shuffle(I, J_nd)
                        k_set = [sig_word_to_idx(K, (n+1)*dim) for K in K_set]
                        covs[n][:, i, j] = esigs[n][..., k_set].sum(axis=-1) - esigs[0][..., i] * esigs[0][..., j]
            cov = covs[0]
            for n in range(1, n_max):
                cov += (1 - n/n_max) * (covs[n] + np.swapaxes(covs[n], -2, -1))
    return cov