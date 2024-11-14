import itertools
import math
from typing import Tuple

def count(j: int, I: Tuple[int, ...]) -> int:
    """
    Count occurrences.
    """
    return sum(j == i for i in I)

def W_set(n: int, d: int):
    """
    Generate all words of length n with alphabet 0,...,d-1.
    """
    return list(itertools.product(range(d), repeat=n))

def K_set(n: int, d: int):
    """
    Generate all even words of length n with alphabet 0,...,d-1.
    """
    if n % 2 == 1:
        return {}
    else:
        return list(I for I in W_set(n, d) if all(count(i, I) % 2 == 0 for i in range(d)))

def C_const(I: Tuple[int, ...], d: int):
    n = len(I) // 2
    C_I = math.factorial(n)/math.factorial(2*n)
    for i in range(d):
        n_i = count(i, I)
        if n_i % 2 == 1:
            raise ValueError('C_I is defined only for even word I.')
        C_I *= math.factorial(n_i)/math.factorial(n_i//2)
    return C_I

def J_set(M: int, tot: Tuple[int, ...]):
    if tot == 0:
        return [(0,)*M]
    elif M == 1:
        return [(tot,)]
    else:
        J = []
        for i in range(tot+1):
            for J_ in J_set(M-1, tot-i):
                J.append(J_ + (i,))
        return J