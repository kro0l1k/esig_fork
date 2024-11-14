import numpy as np
from typing import Tuple, List

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
        return [W + (I[-1],) for W in shuffle(I[:-1], J)] + [W + (J[-1],) for W in shuffle(I, J[:-1])]