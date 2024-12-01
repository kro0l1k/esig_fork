# Expected Signature Estimators

This repo provides code for estimating expected signatures 
```math
\phi(T) := \mathbb{E}\Big[S(\mathbb{X})_{[0,T]}\Big]\in T((\mathbb{R}^d)),
```
from a collection of paths  $\mathbb{X}^{\pi, 1}, \ldots, \mathbb{X}^{\pi, n}$. The paths may be obtained by chopping-and-shifting a single long observation.

We implement both the naive estimator 
```math
\hat{\phi}^{N, \pi}(T) := \frac{1}{N} \sum_{n=1}^N S\left(\mathbb{X}^{n, \pi}\right)_{[0,T]},
```
and the martingale-corrected estimator 
```math
\hat{\phi}^{N, \pi, c}(T) := \frac{1}{N} \sum_{n=1}^N S\left(\mathbb{X}^{n, \pi}\right)_{[0,T]} + \hat{c} S^c\left(\mathbb{X}^{n, \pi}\right)_{[0,T]},
```
where $S^c\left(\mathbb{X}^{n, \pi}\right)_{[0,T]}$ is an Ito correction.

The code is compatible with `numpy` arrays and `torch` tensors, using `iisignature` and `signatory` for signature computations.

These functions can be used directly into more general ML pipelines/models, as illustrated in the forks:
- [conditional-sig-wasserstein-gans](https://github.com/lorenzolucchese/conditional-sig-wasserstein-gans)
- [distribution-regression-streams](https://github.com/lorenzolucchese/distribution-regression-streams)
- [gp-esig-classifier](https://github.com/lorenzolucchese/gp-esig-classifier)
