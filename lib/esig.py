import signatory
import iisignature
import torch
import numpy as np
from typing import overload, List, Union

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


def _expected_signature_estimate_torch(path: torch.Tensor, depth: int, martingale_indices: List[int] = None, stream = False) -> torch.Tensor:    
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
    return signatures.mean(dim=sample_dim)


def _expected_signature_estimate_numpy(path: np.ndarray, depth: int, martingale_indices: List[int] = None, stream = False) -> np.ndarray:    
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
    return signatures.mean(axis=sample_axis)

@overload
def expected_signature_estimate(path: torch.Tensor, depth: int, martingale_indices: List[int] = None, stream = False) -> torch.Tensor: ...

@overload
def expected_signature_estimate(path: np.ndarray, depth: int, martingale_indices: List[int] = None, stream = False) -> np.ndarray: ...

def expected_signature_estimate(path: Union[np.ndarray, torch.Tensor], depth: int, martingale_indices: List[int] = None, stream = False) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(path, torch.Tensor):
        return _expected_signature_estimate_torch(path=path, depth=depth, martingale_indices=martingale_indices, stream=stream)
    elif isinstance(path, np.ndarray):
        return _expected_signature_estimate_numpy(path=path, depth=depth, martingale_indices=martingale_indices, stream=stream)
    else:
        raise ValueError(f'Only torch.Tensor and np.ndarray types supported for path, got = {type(path)}.')