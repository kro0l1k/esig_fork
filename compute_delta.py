import jax
import jax.numpy as jnp
from jax import config
import numpy as np
import signax
import scipy
import warnings
from functools import partial
import os
from sample_different_S import SignalGenerator
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import FastICA
from SOBI import sobi_algo
from statsmodels.tsa.stattools import adfuller

from numpy.linalg import svd, eigh, qr, norm, cond



print("\n\nAvailable devices:", jax.devices())
device_found = False  # Initialize the variable

try:
    metal_devices = jax.devices("METAL")
    if metal_devices:
        config.update("jax_default_device", metal_devices[0])
        print("Using Metal/Apple GPU for computations")
        device_found = True
except Exception as e:
    print(f"Metal device detection error: {e}")

# If Metal wasn't found/set, try standard CUDA GPU
if not device_found:
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            config.update("jax_default_device", gpu_devices[0])
            print("Using CUDA GPU for computations")
            device_found = True
    except Exception as e:
        print(f"CUDA GPU detection error: {e}")

# Try TPU if GPU and Metal both failed
if not device_found:
    try:
        tpu_devices = jax.devices("tpu")
        if tpu_devices:
            config.update("jax_default_device", tpu_devices[0])
            print("Using TPU for computations")
            device_found = True
    except Exception as e:
        print(f"TPU detection error: {e}")

# Fall back to CPU if no accelerators were found
if not device_found:
    print("No accelerators found, using CPU for computations")


def M_IplusE(W, A):
    d = A.shape[0]
    WA = W @ A
    Id= np.eye(d)
    if WA.shape[0] != d or WA.shape[1] != d:
        raise ValueError("W @ A must be a square matrix of shape (d, d)")
    cost = -np.abs(WA)
    r, c = linear_sum_assignment(cost)
    M = np.zeros_like(WA)
    for i, j in zip(r, c):
        M[i, j] = WA[i, j]
    # Convert JAX arrays to NumPy arrays for better Metal compatibility
    M_np = np.array(M)
    # Use NumPy's matrix inversion instead of JAX's
    if np.linalg.cond(M_np) > 1e3:
        warnings.warn("M_inv nearly singular")
    M_inv = np.linalg.inv(M_np)
    WA_np = np.array(WA)
    E = M_inv @ WA_np - Id
    
    return M, E

def get_rel_err(I_x, A_inv, M, E):
    abs_err = np.linalg.norm(I_x - M @ A_inv, 'fro')
    rel_err = abs_err / np.linalg.norm(M @ A_inv, 'fro')
    return rel_err


# ===== FLEX Joint Diagonaliser for Real Matrices =====
class FlexJDReal:
    """
    Joint diagonaliser restricted to real orthogonal matrices whose columns
    have unit l2-norm.  The ``fit`` method returns (W, success_bool, info_dict).
    """
    def __init__(self, max_iter=10_000_000, tol_angle=1e-12,
                 offdiag_tol=1e-6, norm_tol=1e-2):
        self.max_iter   = max_iter       # angular convergence test
        self.tol_angle  = tol_angle
        self.offdiag_tol = offdiag_tol   # diag success test
        self.norm_tol   = norm_tol       # ‖col‖≃1 test

    def fit(self, mats):
        """
        Parameters
        ----------
        mats : list[ndarray]      # each (d,d), typically symmetric
        Returns
        -------
        W         : (d,d) real orthogonal matrix
        success   : bool
        info      : dict   {'iters':…, 'offdiag':…, 'colnorms':…}
        """
        d = mats[0].shape[0]
        # -- real random orthogonal initialisation --------------------
        W, _ = qr(np.random.randn(d, d))            # real    instead of complex
        prev_W = np.empty_like(W)

        # -- main loop ------------------------------------------------
        logging_freq = 10000
        for it in range(self.max_iter):
            prev_W[:] = W
            for i in range(d):
                W_bar = np.delete(W, i, axis=1)
                Q_i   = self._Q_i(mats, W, W_bar)
                C_i   = self._null_space(W_bar.T)    # transpose    instead of conjugate transpose
                C_t   = C_i @ C_i.T
                W[:, i] = self._moqo(C_t, Q_i)       # already unit-normed
            
            # -- check convergence --------------------------------------
            if it % logging_freq == 0:
                offdiag = self._contrast(mats, W)
                col_norms = norm(W, axis=0)
                print(f"Iter {it}: offdiag={offdiag:.2e}, col_norms={col_norms}")
            if np.allclose(W, prev_W, rtol=self.tol_angle):
                break

        # ------------- SUCCESS DIAGNOSTICS ---------------------------
        col_norms  = norm(W, axis=0) 
        offdiag    = self._contrast(mats, W)

        success = (np.all(np.abs(col_norms - 1) < self.norm_tol)
                   and offdiag < self.offdiag_tol)

        info = dict(iters=it+1, offdiag=offdiag, colnorms=col_norms)

        return W, success, info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _Q_i(mats, W, W_bar):
        """Equation (2.4) in the FLEX paper – but *purely real*."""
        Q = np.zeros((W.shape[0], W.shape[0]))
        for R in mats:
            t1 = R @ W_bar @ W_bar.T @ R.T
            t2 = R.T @ W_bar @ W_bar.T @ R
            Q += t1 + t2
        return Q

    @staticmethod
    def _null_space(A, rcond=None):
        """Real null-space (columns form an orthonormal basis)."""
        U, s, Vt = svd(A, full_matrices=True)
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(A.shape)
        rank = (s > rcond * s[0]).sum()
        return Vt[rank:].T

    def _moqo(self, C_tilde, Q):
        """
        Maximise the quartic objective on the *real* Stiefel-1 manifold.
        All inputs are real symmetric, so eigenvectors are real.
        """
        eigvals, eigvecs = eigh(Q)
        thresh = self.tol_angle * np.max(np.abs(eigvals))
        zero   = np.abs(eigvals) < thresh

        if not np.any(zero):                     # Q invertible
            #_, v = eigh(C_tilde, Q)
            _, v = scipy.linalg.eigh(C_tilde, Q)
            w = v[:, -1]
        else:                                    # Q singular  -> cases 2–3
            U0, U1 = eigvecs[:, zero], eigvecs[:, ~zero]
            L1     = eigvals[~zero]
            if U0.size > 0 and not np.allclose(U0.T @ C_tilde @ U0, 0):
                _, v = eigh(U0.T @ C_tilde @ U0)
                w = U0 @ v[:, -1]
            else:
                #_, v = eigh(U1.T @ C_tilde @ U1, np.diag(L1))
                _, v = scipy.linalg.eigh(U1.T @ C_tilde @ U1, np.diag(L1))
                w = U1 @ v[:, -1]

        # enforce unit ℓ₂-norm
        return w / norm(w)

    @staticmethod
    def _contrast(mats, W):
        """Sum of off-diagonal Frobenius norms ‖diag-removed‖²."""
        acc = 0.0
        for R in mats:
            M = W.T @ R @ W
            acc += norm(M - np.diag(np.diag(M)), 'fro')**2
        return acc
from jax import vmap, jit, lax

# ---------- Tridiagonal solve (Thomas algorithm, no triangular_solve) ----------
def _solve_tridiag_thomas(a, b, c, d):
    """
    Solve Ax=d for tridiagonal A with lower diag a, main diag b, upper diag c.
    Shapes: (m,), (m,), (m,), (m,)
    Assumes: a[0]=0, c[-1]=0
    """
    m = b.shape[0]
    # forward sweep
    cprime = jnp.zeros_like(c)
    dprime = jnp.zeros_like(d)

    cprime = cprime.at[0].set(c[0] / b[0])
    dprime = dprime.at[0].set(d[0] / b[0])

    def fwd(i, vals):
        cpr, dpr = vals
        denom = b[i] - a[i] * cpr[i - 1]
        cpr = cpr.at[i].set(jnp.where(i < m - 1, c[i] / denom, 0.0))
        dpr = dpr.at[i].set((d[i] - a[i] * dpr[i - 1]) / denom)
        return (cpr, dpr)

    cprime, dprime = lax.fori_loop(1, m, fwd, (cprime, dprime))

    # back substitution
    x = jnp.zeros_like(d)
    x = x.at[m - 1].set(dprime[m - 1])

    def bwd(i, x_):
        idx = m - 2 - i
        xi = dprime[idx] - cprime[idx] * x_[idx + 1]
        return x_.at[idx].set(xi)

    x = lax.fori_loop(0, m - 1, bwd, x)
    return x

# ---------- Natural cubic spline (equal spacing), pure JAX ----------
@partial(jit, static_argnums=(1,))
def _natural_cubic_spline_1d(y: jnp.ndarray, factor: int = 3) -> jnp.ndarray:
    """
    y: (T,)
    -> (T_ref,) with T_ref = (T-1)*factor + 1
    Requires T >= 3.
    """
    n = y.shape[0]
    assert n >= 3, "Need ≥3 points for spline."

    m = n - 2
    rhs = 3.0 * (y[2:] - 2.0 * y[1:-1] + y[:-2])  # (m,)

    # build tri-diags with padding
    a = jnp.concatenate([jnp.array([0.], y.dtype), jnp.ones((m - 1,), y.dtype)])  # (m,)
    b = 4.0 * jnp.ones((m,), y.dtype)
    c = jnp.concatenate([jnp.ones((m - 1,), y.dtype), jnp.array([0.], y.dtype)])  # (m,)

    m_inner = _solve_tridiag_thomas(a, b, c, rhs)            # (m,)
    m_all = jnp.concatenate([jnp.zeros((1,), y.dtype), m_inner, jnp.zeros((1,), y.dtype)])  # (n,)

    # coefficients on each [i,i+1], h=1
    a0 = y[:-1]
    b0 = (y[1:] - y[:-1]) - (2.0 * m_all[:-1] + m_all[1:]) / 3.0
    c0 = 0.5 * m_all[:-1]
    d0 = (m_all[1:] - m_all[:-1]) / 6.0

    t = jnp.linspace(0.0, 1.0, factor, endpoint=False)[None, :]  # (1,factor)
    vals = a0[:, None] + b0[:, None] * t + c0[:, None] * t**2 + d0[:, None] * t**3  # (n-1,factor)
    vals = vals.reshape(-1)
    vals = jnp.concatenate([vals, y[-1:]], axis=0)
    return vals

@partial(jit, static_argnums=(1,))
def refine_paths_with_spline(paths: jnp.ndarray, factor: int = 2) -> jnp.ndarray:
    """
    paths: (B,T,D) -> (B,T_ref,D)
    """
    def refine_one(path):  # (T,D)
        ys = path.T
        refined = vmap(_natural_cubic_spline_1d, in_axes=(0, None))(ys, factor)  # (D,T_ref)
        return refined.T
    return vmap(refine_one, in_axes=0)(paths)

class SignatureComputer:
    def __init__(self, n, d, MC_SAM_LEN):
        self.n = n
        self.d = d
        self.MC_SAM_LEN = MC_SAM_LEN

    def _lev_2_mat(self, sig2: jnp.ndarray) -> np.ndarray:
        # (d*d,) → (d, d)
        return np.array(sig2).reshape(self.d, self.d)

    def _lev_3_mat(self, sig3: jnp.ndarray) -> np.ndarray:
        # (d*d*d,) → (d, d, d)
        return np.array(sig3).reshape(self.d, self.d, self.d)

    def get_jpaths(self, S: np.ndarray, batch_size: int):
        """
        Split S into subpaths of length MC_SAM_LEN (plus leading zero),
        center if desired, and return as a generator yielding JAX arrays (batch, T, d).
        """
        n_sub = self.n // self.MC_SAM_LEN
        S = S[: n_sub * self.MC_SAM_LEN]
        paths = S.reshape(n_sub, self.MC_SAM_LEN, self.d)

        if self.MC_SAM_LEN > 1:
            paths = paths - paths.mean(axis=0, keepdims=True)

        num_batches = (n_sub + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_sub)
            batch_paths = paths[start_idx:end_idx]
            # center the paths by substracting the mean
            if self.MC_SAM_LEN > 1:
                batch_paths = batch_paths - batch_paths.mean(axis=1, keepdims=True)

            zeros = np.zeros((batch_paths.shape[0], 1, self.d))
            batch_paths_with_zeros = np.concatenate([zeros, batch_paths], axis=1) # (batch, T, d), T = MC_SAM_LEN+1
            yield jnp.array(batch_paths_with_zeros) # Yield JAX array batch


    # ------------------ NEW: refine every subpath before signature ------------------
    def _refine_subpaths(self, jpaths: jnp.ndarray, factor: int = 3) -> jnp.ndarray:
        """
        jpaths: (batch, T, d) → spline-refined (batch, T_ref, d)
        """
        return refine_paths_with_spline(jpaths, factor)


    def compute_lvl2_expected_sig(self, S: np.ndarray, batch_size: int, test_mean_stationarity: bool, refining_with_spline: bool = False) -> np.ndarray:
        """
        Computes the expected level-2 signature for a given signal S using batch processing.
        :param S: Signal of shape (n, d)
        :param batch_size: The size of each batch for processing.
        :return: Expected level-2 signature of shape (d, d)
        """
        n_sub = self.n // self.MC_SAM_LEN
        total_lvl2_sig = jnp.zeros(self.d**2)

        one_sig = jax.jit(lambda path: signax.signature(path, 2))

        for jpaths_batch in self.get_jpaths(S, batch_size):
            if refining_with_spline:
                jpaths_batch = self._refine_subpaths(jpaths_batch, factor=3)

            batched_sigs = jax.vmap(one_sig)(jpaths_batch)  # (batch, channels)
            total_lvl2_sig += batched_sigs[:, self.d : self.d + self.d**2].sum(axis=0)

        avg_lvl2 = total_lvl2_sig / n_sub
        return self._lev_2_mat(avg_lvl2)


    def compute_up_to_lvl3_expected_sig(self, S: np.ndarray, batch_size: int, test_mean_stationarity: bool, mean_stationarity_plot: bool = False, refining_with_spline: bool = False) -> np.ndarray:
        """
        Computes the expected signature for a given signal S of levels 2 and 3 using batch processing.
        :param S: Signal of shape (n, d)
        :param batch_size: The size of each batch for processing.
        :return: Expected signature of shape (d+1, d, d)
        """
        n_sub = self.n // self.MC_SAM_LEN

        total_lvl1_sig = jnp.zeros(self.d)
        total_lvl2_sig = jnp.zeros(self.d**2)
        total_lvl3_sig = jnp.zeros(self.d**3)

        one_sig = jax.jit(lambda path: signax.signature(path, 3))

        # Collect samples for mean stationarity plotting/testing across batches
        mean_stationarity_samples = []
        max_samples_for_plot = 100 # Limit samples for plotting

        for jpaths_batch in self.get_jpaths(S, batch_size):
            if refining_with_spline:
                jpaths_batch = self._refine_subpaths(jpaths_batch, factor=3)

            batched_sigs = jax.vmap(one_sig)(jpaths_batch) # (batch, channels)

            total_lvl1_sig += batched_sigs[:, :self.d].sum(axis=0)
            total_lvl2_sig += batched_sigs[:, self.d : self.d + self.d**2].sum(axis=0)
            total_lvl3_sig += batched_sigs[:, self.d + self.d**2 :].sum(axis=0)

            if mean_stationarity_plot or test_mean_stationarity:
                # Collect a limited number of samples from each batch for analysis
                if len(mean_stationarity_samples) < max_samples_for_plot:
                    samples_to_collect = min(batched_sigs.shape[0], max_samples_for_plot - len(mean_stationarity_samples))
                    mean_stationarity_samples.append(batched_sigs[:samples_to_collect])

        # Concatenate collected samples for stationarity analysis
        if mean_stationarity_samples:
            mean_stationarity_samples = jnp.concatenate(mean_stationarity_samples, axis=0)


        if mean_stationarity_plot:
            total_samples_to_plot = mean_stationarity_samples.shape[0]
            max_value_to_plot = 10.0

            def index_to_multiindex(index, d):
                if index < d:
                    return index+1
                if index < d**2 + d:
                    index -= d
                    # convert (i *d + j) to (i, j)
                    return index // d + 1, index % d + 1
                index -= d**2 + d
                # convert (i *d**2 + j * d + k) to (i, j, k)
                return index // (d * d) + 1, (index % (d * d)) // d + 1, index % d + 1

            # Create y-axis labels with multiindex strings
            y_labels = []
            y_positions = []
            for i in range(mean_stationarity_samples.shape[1]):
                multiindex = index_to_multiindex(i, self.d)
                if isinstance(multiindex, tuple):
                    y_labels.append(str(multiindex))
                else:
                    y_labels.append(f"({multiindex},)")
                y_positions.append(i)

            # Create figure and axes explicitly
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a color gradient from red to blue
            colors = plt.cm.viridis(np.linspace(0, 1, total_samples_to_plot))  # Viridis gradient
            for i in range(mean_stationarity_samples.shape[1]):
                for j in range(total_samples_to_plot):
                    ax.scatter(mean_stationarity_samples[j, i], i, color=colors[j], alpha=0.7, s=20)

            # mark also the mean of the whole mean_stationarity_samples[:, i] for each i
            for i in range(mean_stationarity_samples.shape[1]):
                ax.scatter(mean_stationarity_samples[:, i].mean(), i, marker='x', color='black', s=100)

            # plot only the x values between -3 and 3
            ax.set_xlim(-max_value_to_plot, +max_value_to_plot)
            ax.set_ylim(-1, mean_stationarity_samples.shape[1])
            ax.set_title(f'First {total_samples_to_plot} samples of each signature dimension (purple -> yellow gradient shows temporal order)\nBlack x marks denote the means over the entire signal')

            # Set custom y-axis labels with multiindex
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)

            ax.set_xlabel('Value')
            ax.set_ylabel('Signature index')
            ax.grid()

            # Add a colorbar to show the gradient mapping
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=total_samples_to_plot-1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            # cbar.set_label(f'Sample index (0=purple, {total_samples_to_plot-1}=yellow)')

            plt.show()


        if test_mean_stationarity:
            # for each dimension, test if the mean is stationary
            for i in range(mean_stationarity_samples.shape[1]):
                series = np.array(mean_stationarity_samples[:, i]) # Convert JAX array to NumPy for statsmodels
                is_stationary, adf_stat, adf_p = test_mean_stationarity_adf(series)
                if not is_stationary:
                    warnings.warn(f"####Series {i} is not stationary: ADF stat={adf_stat}, p-value={adf_p}#####")
                else:
                    if adf_p > 0.001:
                        print(f"Series {i} is stationary, but we had a non-zero p value: ADF stat={adf_stat}, p-value={adf_p}")


        avg_lvl2 = total_lvl2_sig / n_sub
        avg_lvl3 = total_lvl3_sig / n_sub

        M_only_lvl2 = self._lev_2_mat(avg_lvl2.ravel())
        M_only_lvl2 = np.expand_dims(M_only_lvl2, axis=0)  # (1, d, d)
        M_only_lvl3 = self._lev_3_mat(avg_lvl3.ravel())  # (d, d, d)
        M_2and3 = np.concatenate([M_only_lvl2, M_only_lvl3], axis=0)

        return M_2and3
def check_identifiability(S, batch_size, kappa_thresh=0.01, mc_sam_len=None):
    # this can take a while for large n

    n, d = S.shape
    # Use the provided MC_SAM_LEN from the caller or default to the one passed in
    actual_mc_sam_len = mc_sam_len if mc_sam_len is not None else 5
    to_check = SignatureComputer(n, d, actual_mc_sam_len)
    M2and3 = to_check.compute_up_to_lvl3_expected_sig(S, batch_size, test_mean_stationarity=False)
    Monly2 = M2and3[0, :, :]  # (d, d)
    Monly3 = M2and3[1:, :, :]  # (d,d,d)
    zeros = sum(abs(Monly3[k, k, k]) < kappa_thresh for k in range(d))
    if zeros > 1:
        warnings.warn("Identifiability violation: >1 zero third-moments")
        print("The diagonal third-moments were: ", [Monly3[k, k, k] for k in range(d)])
        print("The number of zero third-moments is: ", zeros)

    return


class ContrastCalculator:
    def __init__(self, n, d, MC_SAM_LEN, batch_size, check_identifiability_criteria=True, show_mean_stationarity_plot=False, verbose=False):
        self.n = n
        self.d = d
        self.MC_SAM_LEN = MC_SAM_LEN
        self.batch_size = batch_size
        self.signaturecomputer = SignatureComputer(n, d, MC_SAM_LEN)
        self.check_identifiability_criteria = check_identifiability_criteria
        self.verbose = verbose
        self.show_mean_stationarity_plot = show_mean_stationarity_plot  # Set to True if you want to plot mean stationarity

    def compute_N_fromM2(self, M2: np.ndarray) -> np.ndarray:
        """takes the signal nu and computes the N matrix, which is diagonal with entries sqrt<mu>_ii

        Args:
            M2 (np.ndarray): signal of shape (d, d)

        Returns:
            np.ndarray: N matrix of shape (d, d)
        """
        diag = np.sqrt(np.diag(M2))
        res = np.diag(diag)
        if np.linalg.cond(res) > 1e3:
            warnings.warn("N nearly singular")
            print("N matrix is: \n", res)
            print("M2 matrix is: \n", M2)
        return res

    def get_Mu_star_matrices(self, Mu_matrices: np.ndarray) -> np.ndarray:
        """returns the Mu_star matrices, which are the Mu matrices in the perfectly IC case.

        Args:
            Mu_matrices (np.ndarray): shape (d+1, d, d)

        Returns:
            np.ndarray: shape (d+1, d, d)
        """
        Mu_only2 = Mu_matrices[0]  # (d, d)
        Mu_only3 = Mu_matrices[1:]  # (d, d, d)
        Mu_star = np.zeros_like(Mu_matrices)
        Mu_star[0] = np.diag(np.diag(Mu_only2))  # D[0] is the diagonal of M2
        # for level 3
        for k in range(0, self.d):
            D = np.zeros_like(Mu_only3[k])
            D[k, k] = Mu_only3[k, k, k]
            Mu_star[k + 1] = D
        return Mu_star

    def compute_delta(self, S:np.ndarray) -> float:
        """computes the IC-defect delta for the sources S.

        Args:
            S (np.ndarray): shape(n, d)

        Returns:
            float: delta (IC-defect)
        """

        if self.check_identifiability_criteria:
            check_identifiability(S, self.batch_size, mc_sam_len=self.MC_SAM_LEN)
        Mu_matrices = self.signaturecomputer.compute_up_to_lvl3_expected_sig(S, self.batch_size, test_mean_stationarity=False, mean_stationarity_plot=self.show_mean_stationarity_plot)

        print("Coredinates (Mu_matrices) of the sources S are: \n", Mu_matrices)

        N = self.compute_N_fromM2(Mu_matrices[0])
        eps = 1e-16
        if np.linalg.cond(N) > 1e3:
            warnings.warn("N nearly singular")
            print("N matrix is: \n", N)
            print("Mu matrices are: \n", Mu_matrices)
        # add eps * Id to N to make it invertible
        N = N + eps * np.eye(N.shape[0])
        Ninv = np.linalg.inv(N)

        denominator = [1.0]
        denominator.extend(np.diag(N))
        denominator = np.array(denominator)  # (d+1,)

        Mu_star_matrices = self.get_Mu_star_matrices(Mu_matrices)

        difference = Mu_matrices - Mu_star_matrices

        sum_sq = 0.0
        for k in range( self.d + 1):
            k_th_component = np.linalg.norm(Ninv @ (difference[k] / denominator[k]) @ Ninv, 'fro')**2
            sum_sq += k_th_component
        res = np.sqrt(sum_sq)
        if self.verbose:
            print(" DEBUG FOR IC-defECT CALCULATOR")
            print(" Mu matrices while computing delta (should be diagonal after 3.12.): \n", Mu_matrices)
            print(" delta: ", res)
            print(" Mu matrices for the sources S are: (check visually if they are close to diagonal) \n", Mu_matrices)
            print(" END DEBUG FOR IC-defECT CALCULATOR")

        return res

class Optimizer:
    def __init__(self, n, d, MC_SAM_LEN, batch_size, verbose=False):
        self.n = n
        self.d = d
        self.MC_SAM_LEN = MC_SAM_LEN
        self.batch_size = batch_size
        self.signature_computer = SignatureComputer(n, d, MC_SAM_LEN)
        self.contrast_calculator = ContrastCalculator(n, d, MC_SAM_LEN, batch_size)
        self.verbose = verbose

    def inverse_sqrt_psd_matrix(self, M_psd: np.ndarray, eps=1e-12) -> np.ndarray:
        """Computes the inverse square root of a positive semi-definite matrix M.
        """
        M_sym = (M_psd + M_psd.T)/2
        w, v = np.linalg.eigh(M_sym)
        w = np.maximum(w, eps)
        result = (v * (1/(np.sqrt(w) + eps))) @ v.T
        if not np.allclose(result @ result @ M_psd, np.eye(M_psd.shape[0]), atol=1e-3):
            warnings.warn("Inverse square root did not find the right matrix.")
            print(" M_psd: \n", M_psd)
            print(" result: \n", result)
        return result

    def compute_R_fromM2(self, M2: np.ndarray) -> np.ndarray:
        """takes the second signature moments and returns the whitenting matrix R.
        Args:
            M2 (np.ndarray): second order coredinates of shape (d, d)

        Returns:
            np.ndarray: R matrix of shape (d, d)
        """
        C = 0.5*(M2 + M2.T)
        # C_np = np.array(C)
        if np.linalg.cond(C) > 1e3:
            print(" we are trying to compute R from the M2 matrix, but C is nearly singular.")
            warnings.warn("C nearly singular")
            print("C (the symmetric part of M2) matrix is: \n", C)
            print("M2 matrix is: \n", M2)
        R = self.inverse_sqrt_psd_matrix(C)

        return np.array(R)

    def contrast_from_Mu(self, Mu_matrices: np.ndarray) -> float:
        """Computes the contrast from the Mu matrices.

        Args:
            Mu_matrices (np.ndarray): Mu matrices of shape (d+1, d, d)

        Returns:
            float: Contrast value
        """
        sum_sq = 0.0
        for k in range(0, self.d + 1):
            M_no_diag = Mu_matrices[k] - np.diag(np.diag(Mu_matrices[k]))
            k_th_component = np.linalg.norm(M_no_diag, 'fro')**2
            sum_sq += k_th_component

        return np.sqrt(sum_sq)

    def phi_from_signal(self, theta, R, X) -> float:

        thetaR = theta @ R  # (d, d)
        thetaR_X = X @ thetaR.T  # (n, d)
        Mu_thetaRX = self.signature_computer.compute_up_to_lvl3_expected_sig(thetaR_X, self.batch_size, test_mean_stationarity=False)
        if self.verbose:
            print("\n debugging phi_from_signal")
            print(" Mu_thetaRX: ", Mu_thetaRX)
        Mu2 = Mu_thetaRX[0]  # (d, d)
        Mu3 = Mu_thetaRX[1:]  # (d, d, d)
        N = self.contrast_calculator.compute_N_fromM2(Mu2)
        Ninv = np.linalg.inv(N)
        denominator = np.diag(N)
        x_stats = np.zeros((self.d + 1, self.d, self.d))
        x_stats[0] = Ninv @ (Mu2 - np.diag(np.diag(Mu2))) @ Ninv  # level 2
        for k in range(0, self.d):
            x_stats[k + 1] = Ninv @ ((Mu3[k] - np.diag(np.diag(Mu3[k]))) / denominator[k]) @ Ninv

        phi2 = 0.0
        for k in range(0, self.d + 1):
            phi2 += np.linalg.norm(x_stats[k], 'fro')**2
            if self.verbose:
                print(f"  k = {k}, norm of x_stats[k]: {np.linalg.norm(x_stats[k], 'fro')}")
        phi = np.sqrt(phi2)

        return phi

    def compute_x_statistics(self, thetaX: np.ndarray) -> np.ndarray:
        """Computes the x-statistics for the whitened signal X.
        Args:
            X (np.ndarray): shape (n, d)

        Returns:
            np.ndarray: x-statistics of shape (d+1, d, d)
        """
        M2uand3 = self.signature_computer.compute_up_to_lvl3_expected_sig(thetaX, self.batch_size, test_mean_stationarity=False)
        Mu_only2 = M2uand3[0]  # (d, d)
        Mu_only3 = M2uand3[1:]  # (d, d, d)


        N = self.contrast_calculator.compute_N_fromM2(Mu_only2)
        Ninv = np.linalg.inv(N)

        denominator = np.diag(N)
        x_stats = np.zeros((self.d + 1, self.d, self.d))
        x_stats[0] = Ninv @ Mu_only2 @ Ninv  # level 2
        for k in range(0, self.d):
            x_stats[k + 1] = Ninv @ (Mu_only3[k] / denominator[k]) @ Ninv
        if self.verbose:
            print(" DEBUG FOR OPTIMIZER")
            print(" the x_statistics are: ", x_stats)
            print(" N matrix is: \n", N)
            print(" M2uand3: \n", M2uand3)
            print(" for the above matrices, the value of the contrast is: ", self.contrast_from_Mu(M2uand3))
            print(" END DEBUG FOR OPTIMIZER")
        return x_stats

    def RICA(self, X: np.ndarray) -> np.ndarray:
        """Performs RICA on the signal X.
        Args:
            X (np.ndarray): shape (n, d)
        Returns:
            np.ndarray: RICA unmixing matrix W_x of shape (d, d)
        """

        M2 = self.signature_computer.compute_lvl2_expected_sig(X, self.batch_size, test_mean_stationarity=False)
        # print(" second moments of the observed signal X: \n", M2)

        # if M2 is close to being the 0 matrix, print out the signature up to level 3
        if np.linalg.norm(M2, 'fro') < 1e-6:
            print(" M2 is close to being the zero matrix, printing out the signature up to level 3:")
            Mu_matrices = self.signature_computer.compute_up_to_lvl3_expected_sig(X, self.batch_size, test_mean_stationarity=False)
            print(" Mu matrices: \n", Mu_matrices)

            # plot the first 10000 samples of each dimension of the signal X
            for i in range(self.d):
                plt.plot(X[:10000, i], label=f'dim {i+1}')
            plt.title('First 10000 samples of each dimension of the signal X')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()
            warnings.warn("M2 is close to being the zero matrix, check your signal X.")

        R = self.compute_R_fromM2(M2)
        if self.verbose:
            print(" R matrix: \n", R)

        X_whitened = np.dot(X, R.T)

        x_statistics = self.compute_x_statistics(X_whitened)

        if self.verbose:
            print(" x statistics: \n", x_statistics)

        jd = FlexJDReal(max_iter=1000_000, tol_angle=1e-12, offdiag_tol=1e-4, norm_tol=1e-4)
        V, ok, info = jd.fit(list(x_statistics))

        theta_hat = V.T  # we should transpose, because FlexJD returns V s.t. V.T [M] V = D and we want theta [M] theta.T = D
        print(" condition number of theta_hat: ", cond(theta_hat))
        print(" norms of rows of theta_hat: ", norm(theta_hat, axis=1))
        W_x = theta_hat @ R
        if not ok:
            warnings.warn(
                f"FlexJD failed: off-diag={info['offdiag']:.2e}, "
                f"row norms of theta_hat={info['colnorms']}"
            )
        if self.verbose:
            print("FlexJD converged in", info['iters'], "iterations")
            print("Off-diagonal norm:", info['offdiag'])
            print("Row norms of theta_hat:", info['colnorms'])
            print("Condition number of theta_hat:", cond(theta_hat))
        return W_x


