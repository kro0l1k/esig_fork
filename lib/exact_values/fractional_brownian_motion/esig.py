from typing import Union, Tuple
import math
import numpy as np
import itertools
import sympy
import scipy.integrate
import warnings

def expected_signature_fBM(j: Union[int, Tuple[int, ...]], T: float, d: int, H: float, iterated_integrals: Union[dict, None] = None) -> Union[np.ndarray, float]:
    """
    Theorem 31 in F. Baudoin, L. Coutin / Stochastic Processes and their Applications 117 (2007) 550-574.
    """
    assert 0.5 < H < 1, "Formula only valid for Hurst exponent in (1/2, 1)." 
    if isinstance(j, int):
        if j == 0:
            return 1
        elif j > 4:
            warnings.warn('This will take a long time...')
        rho = np.zeros((d,)*j)
        if j % 2 == 0:
            k = j // 2
            iterated_integrals = {}
            for perm in symmetric_group(2*k):
                iterated_integrals[perm] = iterated_integral(perm, H)
            for I in np.ndindex((d,)*j):
                rho[I] = expected_signature_fBM(I, T, d, H, iterated_integrals=iterated_integrals)
        return rho
    elif isinstance(j, tuple):
        assert all([0 <= i < d for i in j]), f"Word {j} does not match dimension d={d}"
        if j == tuple():
            return 1.
        if len(j) % 2 == 0:
            k = len(j) // 2
            sum = 0
            for perm in symmetric_group(2*k):
                print(perm)
                kronecker_delta = 1
                for l in range(k):
                    kronecker_delta *= (j[perm[2*(l+1) - 1]] == j[perm[2*(l+1) - 2]])
                if kronecker_delta == 1:
                    sum += iterated_integrals[perm] if iterated_integrals else iterated_integral(perm, H)
                else:
                    continue
            return T**(2*k*H) * H**k * (2*H - 1)**k / (math.factorial(k) * 2**k) * sum
        else:
            return 0.
    else:
        raise ValueError('Can only compute expected signature of word (tuple) or signature level (int).')
    

def symmetric_group(k: int) -> itertools.permutations:
    elements = list(range(k))
    permutations = itertools.permutations(elements)
    return permutations


def iterated_integral(perm: Tuple[int, ...], H: float, integration_method: str = "symbolic"):
    k = len(perm)
    assert k % 2 == 0, "The iterated integral is defined only for even length permutations."
    def f(t):
        prod = 1.
        for l in range(k // 2):
            prod *= np.abs(t[perm[2*(l+1) - 1]] - t[perm[2*(l+1) - 2]])**(2*H - 2)
        return prod
    
    if integration_method == "symbolic":
        s = sympy.symbols(" ".join([f"s{i}" for i in range(k)]), positive=True)
        from sympy import Q, refine, simplify
        constraints = sympy.And(*[s[i] < s[i+1] for i in range(k-1)], s[k-1] < 1)
        func = f(s)
        for i in range(k - 1):
            integral = sympy.Integral(func, (s[i], 0, s[i + 1]))
            func = simplify(refine(replace_min(integral.doit()), constraints))
        return sympy.integrate(func, (s[k - 1], 0, 1)).evalf()
    
    elif integration_method == "numeric":
        def bounds(*t, i):
            if i < k - 1:
                if len(t) < i + 1:
                    return (0, 1)
                else:
                    return (0, t[i])
            elif i == k - 1:
                return (0, 1)
            
        from functools import partial
        ranges = []
        for i in range(k):
            ranges.append(
                partial(bounds, i=i) 
            )

        def f_unpacked(*t):
            return f(t)
        return scipy.integrate.nquad(f_unpacked, ranges, opts={'epsabs': 100})[0]
    

def replace_min(expr):
    if expr.func == sympy.Min:
        terms = expr.args
        piecewise_expr = sympy.Piecewise(
            *[(arg, arg <= min(other for other in terms if other != arg)) for arg in terms]
        )
        return piecewise_expr
    elif expr.args:
        return expr.func(*[replace_min(arg) for arg in expr.args])
    return expr