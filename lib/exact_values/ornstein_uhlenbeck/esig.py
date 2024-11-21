from typing import Union, Tuple, List
import numpy as np
import sympy
import sympy.stats
from lib.data.utils import MCAR_A


def expected_signature_OU(j: Union[int, Tuple[int, ...]], T: float, A: np.ndarray, Sigma: np.ndarray) -> Union[np.ndarray, float]:
    """
    We assume dX = -AX dt + Sigma dW is in the stationary regime.
    """
    d = A.shape[0]
    if isinstance(j, int):
        if j == 0:
            return 1
        elif j == 1:
            return np.zeros(d)
        elif j == 2: # or j == 3:
            A = sympy.Matrix(A)
            Sigma = sympy.Matrix(Sigma)
            Id = sympy.Matrix(np.eye(d))
            u = sympy.symbols("u", positive=True)
            Sigma_X = sympy.integrate(sympy.exp(-u*A) @ Sigma @ sympy.exp(-u*A.T), (u, 0, np.inf)) 
            # if j == 2:
            return - Sigma_X @ A.T * T + Sigma_X * (Id - sympy.exp(- A.T * T)) + 0.5 * Sigma * T
            # elif j == 3:
            #     signature = np.zeros((d,)*3)
            #     s, t = sympy.symbols("s t", positive=True)
            #     x = sympy.symbols(" ".join([f"x{i}" for i in range(d)]))

            #     time_integral = sympy.integrate((sympy.exp(-A*s) - Id) @ sympy.Matrix(x) @ sympy.Matrix(x).T @ sympy.exp(-A.T*s) @ A.T, (s, 0, t), (t, 0, T))

            #     for i in np.ndindex((d,)*3):
            #         print(i)
            #         def f(x):
            #             return sympy.tensorproduct(sympy.Matrix(x).T @ A, time_integral)[0, i[0], i[1], i[2]] * (2*sympy.pi)**(-d/2) * Sigma_X.det()**(-0.5) * sympy.exp(- 0.5 * sympy.Matrix(x).T @ Sigma_X.inv() @ sympy.Matrix(x))[0, 0]

            #     signature[i] = sympy.integrate(f(x), *[(x_, -sympy.oo, +sympy.oo) for x_ in x])
            #     return signature
        elif j == 3:
            return np.zeros((d,)*3)
        else:
            raise NotImplementedError(
                "For our purposes, we only implement the first three levels,"
                "the full expected signature for arbitrary initial condition can be computed using H. Ni's PDE approach."
            )
    elif isinstance(j, tuple):
        if j == tuple():
            return 1.
        return expected_signature_OU(len(j), T, A)[j]
    else:
        raise ValueError('Can only compute expected signature of word (tuple) or signature level (int).')
    

def expected_signature_MCAR(j: Union[int, Tuple[int, ...]], T: float, AA: Union[Tuple[Union[float, np.ndarray], ...], List[Union[float, np.ndarray]]], Sigma: Union[np.ndarray, None] = None) -> Union[np.ndarray, float]:
    """
    We assume the driving Brownian motion is driftless and the MCAR process is in the stationary regime.
    """
    p = len(AA)
    d = AA[0].shape[0]
    if Sigma is None:
        Sigma = np.eye(d)
    pd = p*d
    A_OU = MCAR_A(AA)
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_OU = E.dot(Sigma).dot(E.T)
    if isinstance(j, int):
        if j == 0:
            return 1
        else:
            return expected_signature_OU(j, T, A_OU, Sigma_OU)[tuple(slice(d) for _ in range(j))]
    elif isinstance(j, tuple):
        if j == tuple():
            return 1.
        return expected_signature_MCAR(len(j), T, AA, Sigma)[j]
    else:
        raise ValueError('Can only compute expected signature of word (tuple) or signature level (int).')