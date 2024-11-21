import numpy as np
from typing import Tuple, List, Union

def sig_idx_to_word(idx: int, dim: int) -> Tuple[int, ...]:
    depth = 0
    while idx >= 0:
        depth += 1
        idx_level = idx
        idx -= dim**depth
    word = np.unravel_index(idx_level, (dim,)*depth)
    return word


def sig_word_to_idx(word: Tuple[int], dim: int) -> int:
    assert all([(0 <= w) and (w < dim) for w in word])
    idx_start = sum([dim**i for i in range(1, len(word))])
    idx_level = np.ravel_multi_index(word, (dim,)*len(word))
    return idx_start + idx_level


def shuffle(I: Tuple[int, ...], J: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Use iterative definition.
    """
    if I == tuple():
        return [J]
    elif J == tuple():
        return [I]
    else:
        return set([W + (I[-1],) for W in shuffle(I[:-1], J)] + [W + (J[-1],) for W in shuffle(I, J[:-1])])
    

def chop_and_shift(paths: np.ndarray, chops: Union[int, List[int]]) -> Union[np.ndarray, List[np.ndarray]]:
    """
    :param paths: collection of paths of shape (..., length, dim)
    :param chops: points at which to chop the path, if int needs to divide length - 1 and will be broadcasted
    return: chopped_paths: 
            if chops is int: a np.ndarray of shape (..., (length - 1) // chops, chops + 1, dim)
            else: a list of length len(chops) - 1 containing the chopped paths with shapes (..., chops[i+1] + 1 - chops[i], dim)
    """
    length = paths.shape[-2]
    if isinstance(chops, int):
        assert length % chops == 1
        return np.stack(
            chop_and_shift(
                paths, 
                chops = [i*chops for i in range((length - 1) // chops + 1)]
                ),
            axis=-3
            )
    else:
        chopped_paths = []
        for N_start, N_end in zip(chops[:-1], chops[1:]):
            chopped_paths.append(paths[..., N_start:(N_end+1), :] - paths[..., N_start:N_start+1, :])
        return chopped_paths
    

def chop_shift_and_augment(paths: np.ndarray, n: int, chop: int) -> np.ndarray:
    """
    :param paths: collection of paths of shape (..., length, dim)
    :param n: cross-covariance term to be estimated
    """
    length, dim = paths.shape[-2], paths.shape[-1]
    assert n <= (length- 1) // chop - 1, f'Cannot chop, shift and augment paths of shape {paths.shape} when n = {n} and chop = {chop}.'
    X = chop_and_shift(paths, chops=chop)
    new_paths = []
    for m in range(0, (length - 1) // chop - n, n + 1):
        new_path = np.zeros((*X.shape[:-3], (n + 1) * chop + 1, (n + 1) * dim))
        for i in range(n + 1):
            new_path[..., i*chop:(i+1)*chop+1, i*dim:(i+1)*dim] = X[..., m + i, :, :]
            if i < n:
                new_path[..., (i+1)*chop+1:, i*dim:(i+1)*dim] = X[..., m + i, -1:, :]
        new_paths.append(new_path)
    return np.stack(new_paths, axis=-3)