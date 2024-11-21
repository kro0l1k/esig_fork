from typing import Union, Tuple
import numpy as np
import math
from functools import partial
from lib.utils import shuffle
from lib.brownian_motion.utils import J_set, K_set, W_set, C_const

def expected_signature_linear_approx_BM(j: Union[int, Tuple[int, ...]], M: int, T: float, d: int) -> Union[np.ndarray, float]:
    if isinstance(j, int):
        if j == 0:
            return 1
        rho = np.zeros((d,)*j)
        if j % 2 == 0:
            J = J_set(M, j//2)
            for j in J:
                values = {}
                for m in range(M):
                    if j[m] > 0:
                        K = K_set(2*j[m], d)
                        old_values = values
                        values = {}
                        for I in K:
                            if len(old_values) == 0:
                                values[I] = C_const(I, d) * (T / (2*M))**j[m] / math.factorial(j[m])
                            else:
                                for I_old, value_old in old_values.items():
                                    values[I_old + I] = value_old * C_const(I, d) * (T / (2*M))**j[m] / math.factorial(j[m])
                for I, value in values.items():
                    rho[I] += value
        return rho
    elif isinstance(j, tuple):
        if j == tuple():
            return 1
        rho = expected_signature_linear_approx_BM(len(j), M, T, d)
        return rho[j]
    else:
        raise ValueError('Can only compute expected signature of word (tuple) or signature level (int).')
    
def expected_signature_BM(j: Union[int, Tuple[int, ...]], T: float, d: int) -> Union[np.ndarray, float]:
    if isinstance(j, int):
        if j == 0:
            return 1
        rho = np.zeros((d,)*j)
        if j % 2 == 0:
            for I in W_set(j//2, d):
                II = tuple()
                for i in I:
                    II += (i, i)
                rho[II] = expected_signature_BM(II, T, d)
        return rho
    elif isinstance(j, tuple):
        if j == tuple():
            return 1.
        if j[::2] == j[1::2]:
            return (T/2)**(len(j)//2) / math.factorial(len(j)//2)
        else:
            return 0.
    else:
        raise ValueError('Can only compute expected signature of word (tuple) or signature level (int).')

def var_estimator(I: Tuple[int, ...], M: int, T: float, d: int) -> float:
    if M is None:
        expected_signature = expected_signature_BM
    elif isinstance(M, int):
        expected_signature = partial(expected_signature_linear_approx_BM, M=M)
    variance = 0.
    higher_order_terms = expected_signature(j=2*len(I), T=T, d=d)
    for J in shuffle(I, I):
        variance += higher_order_terms[J]
    variance -= expected_signature(j=I, T=T, d=d)**2
    return variance