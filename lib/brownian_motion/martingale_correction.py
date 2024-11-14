import math
import numpy as np

def var_strat(level: int, T: float, M: int = None):
    """
    Variance of Stratonovich signature of a one-dimensional continuous-time Brownian motion and its piecewise-linear approximation (Young/Lebesgue/Riemann integration). 
    """
    s2 = math.factorial(2*level) / (2**level * math.factorial(level)**3) * T**level
    if level%2 == 0:
        return s2 - 1 / (2**level * math.factorial(level//2)**2) * T**level
    else:
        return s2
    
def var_control(level: int, T: float, M: int = None):
    """
    Variance of control. 
    """
    if M is None:
        return math.factorial(2*level - 2) / (2**(level-1) * math.factorial(level) * math.factorial(level - 1)**2) * T**level
    elif isinstance(M, int) and M >= 1:
        return math.factorial(2*level - 2) / (2**(level-1) * math.factorial(level - 1)**3) * (T/M)**level * sum([m**(level-1) for m in range(M)])

def cov_strat_control(level: int, T: float, M: int = None):
    """
    Covariance of signature term and control. 
    """
    if M is None:
        return 1 / math.factorial(level) * (T/2)**level * (math.factorial(2*level) / math.factorial(level)**2 - sum([math.comb(level - 2 + i, i) for i in range(level+1) if (i+level)%2 == 0]))
    elif isinstance(M, int) and M >= 1:
        return 2 / math.factorial(level-1) * (T/(2*M))**level * sum([sum([math.factorial(2*level - 2*j - 2) / (math.factorial(level - 2*j - 1) * math.factorial(level - j - 1) * math.factorial(j)) * (M-m)**j * m**(level-j-1) for j in range(0, (level-1)//2+1)]) for m in range(M)])

def corr_strat_control(level: int, T: float, M: int = None):
    return cov_strat_control(level, T, M) / np.sqrt(var_control(level, T, M) * var_strat(level, T, M))

def loose_lower_bound_corr(level: int):
    bound = 0.5*np.sqrt(2-1/level)
    return bound
    if level%2 == 0:
        adj = 1/np.sqrt(1 - math.factorial(level)**3 / (math.factorial(2*level) * math.factorial(level//2)**2))
        return bound * adj
    else:
        return bound