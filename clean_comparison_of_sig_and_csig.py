import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import signax
import numpy as np

from sample_different_S import SignalGenerator

def get_a_path(n, d):
    assert d == 2
    t = jnp.linspace(0, 4 * jnp.pi, n)
    dim1 = jnp.sin(t)[:, None] + (np.random.randn(n) * 0.1)[:, None]
    dim2 = jnp.cos(t)[:, None] + (np.random.randn(n) * 0.1)[:, None]
    return jnp.concatenate([dim1, dim2], axis=1)

def _tensor(a, b):
    return jnp.einsum("i,j->ij", a, b)

def _tensor3(a, b, c):
    return jnp.einsum("i,j,k->ijk", a, b, c)


@jax.jit
def signature_upto3(path, order: int):
    # if order < 1 or order > 3:
    #     raise ValueError("order must be 1, 2, or 3")

    inc = jnp.diff(path, axis=0)          # (n-1, d)
    d = inc.shape[1]

    S1 = jnp.zeros((d,))
    S2 = jnp.zeros((d, d))
    S3 = jnp.zeros((d, d, d))

    def body(i, state):
        S1, S2, S3 = state
        a = inc[i]
        a2 = 0.5 * _tensor(a, a)
        a3 = (1.0 / 6.0) * _tensor3(a, a, a)

        S1_new = S1 + a
        S2_new = S2 + _tensor(S1, a) + a2
        S3_new = S3 + jnp.einsum("ij,k->ijk", S2, a) + jnp.einsum("i,jk->ijk", S1, a2) + a3
        return (S1_new, S2_new, S3_new)

    S1, S2, S3 = jax.lax.fori_loop(0, inc.shape[0], body, (S1, S2, S3))

    # if order == 1:
    #     return S1
    # if order == 2:
    #     return jnp.concatenate([S1, S2.ravel()])
    return jnp.concatenate([S1, S2.ravel(), S3.ravel()])


@jax.jit
def controlled_signature_upto3(path, order: int):
    """
    Control-variates vector S^c up to 'order' (1..3) in the same flattening
    as 'signature_upto3': [level-1, level-2.ravel, level-3.ravel].
    Implements Eq. (16)/(discrete definition) from §2.2: accumulate
      level-1: 1 * ΔX
      level-2: prefix S1 ⊗ ΔX
      level-3: prefix S2 ⊗ ΔX
    with prefix signatures updated by Chen after each accumulation.
    """
    # if order < 1 or order > 3:
    #     raise ValueError("order must be 1, 2, or 3")

    inc = jnp.diff(path, axis=0)          # (n-1, d)
    d = inc.shape[1]

    # Prefix signatures (start at zero path)
    S1 = jnp.zeros((d,))
    S2 = jnp.zeros((d, d))
    S3 = jnp.zeros((d, d, d))  # kept for completeness; not needed to compute controls ≤3

    # Control accumulators
    C1 = jnp.zeros((d,))
    C2 = jnp.zeros((d, d))
    C3 = jnp.zeros((d, d, d))

    def body(i, state):
        S1, S2, S3, C1, C2, C3 = state
        a = inc[i]
        a2 = 0.5 * _tensor(a, a)
        a3 = (1.0 / 6.0) * _tensor3(a, a, a)

        # --- accumulate controls using prefix BEFORE advancing the segment ---
        C1 = C1 + a                                    # level-1 words: empty-prefix * ΔX
        C2 = C2 + jnp.einsum("i,j->ij", S1, a)         # level-2 words: S1_prefix ⊗ ΔX
        C3 = C3 + jnp.einsum("ij,k->ijk", S2, a)       # level-3 words: S2_prefix ⊗ ΔX

        # --- advance prefix signatures via Chen (exp(a)) ---
        S1_new = S1 + a
        S2_new = S2 + _tensor(S1, a) + a2
        S3_new = S3 + jnp.einsum("ij,k->ijk", S2, a) + jnp.einsum("i,jk->ijk", S1, a2) + a3

        return (S1_new, S2_new, S3_new, C1, C2, C3)

    S1, S2, S3, C1, C2, C3 = jax.lax.fori_loop(0, inc.shape[0], body, (S1, S2, S3, C1, C2, C3))

    # if order == 1:
    #     return C1
    # if order == 2:
    #     return jnp.concatenate([C1, C2.ravel()])
    return jnp.concatenate([C1, C2.ravel(), C3.ravel()])

def compute_c_star(normal_sigs, controlled_sigs):
    # X: (n, p), Y: (n, p)
    X = normal_sigs
    Y = controlled_sigs
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    Yc = Y - jnp.mean(Y, axis=0, keepdims=True)
    # population-style (divide by n); matches jnp.var default (population)
    cov_term = jnp.mean(Xc * Yc, axis=0)            # (p,)
    var_term = jnp.mean(Yc * Yc, axis=0)            # (p,)
    # safe divide
    var_term = jnp.where(var_term == 0, 1.0, var_term)
    return cov_term / var_term


def main():
    n_experimetns, n_samples_per_exp, n, d, order = 20, 100, 500, 2,  3
    # path = get_a_path(n, d)
    signal_generator = SignalGenerator()
    
    
    path = signal_generator.sample_s(d,n,"OU")

    # plot the sampled path
    plt.plot(path)
    plt.title("Sampled Path")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
    plt.close()

    classic_Estimator = []
    control_Variate_Estimator = []

    for exp_idx in range(n_experimetns):    
        sigs_lib = []
        sigs_manual = []
        ctrls_manual = []

        for i in range(n_samples_per_exp):
            path = signal_generator.sample_s(d,n,"OU")
            # print(" Sampled path shape: ", path.shape)
            sig_lib = signax.signature(path, order)  # your installed signax uses (path, order)
            
            sig_manual = signature_upto3(path, order)
            ctrl_manual = controlled_signature_upto3(path, order)

            sigs_lib.append(sig_lib)
            sigs_manual.append(sig_manual)
            ctrls_manual.append(ctrl_manual)

        # for each of the indices, compare the distributions 
        sigs_lib = jnp.stack(sigs_lib)
        sigs_manual = jnp.stack(sigs_manual)
        ctrls_manual = jnp.stack(ctrls_manual)  
    
    
        c_star = compute_c_star(sigs_lib, ctrls_manual)

        print(f"Experiment {exp_idx}: c_star = {c_star}")
        print(" shape of c_star: ", c_star.shape , " should be 14")
        
        MC_estimate = jnp.mean(sigs_lib, axis=0)
        # print(f"Experiment {exp_idx}: MC estimate = {MC_estimate.shape}, should be (14,)    ")
        
        CV_estimate = MC_estimate - c_star * jnp.mean(ctrls_manual, axis=0)
        # print(f"Experiment {exp_idx}: CV estimate = {CV_estimate.shape}, should be (14,)    ")
        classic_Estimator.append(MC_estimate)
        control_Variate_Estimator.append(CV_estimate)
        

    # Convert lists to arrays for plotting
    classic_Estimator = jnp.stack(classic_Estimator)
    control_Variate_Estimator = jnp.stack(control_Variate_Estimator)
    
    # generate plots to compare distributions of estimators
    num_indices = classic_Estimator.shape[1]
    n_cols = 4
    # Limit to max 6 rows
    n_rows = min((num_indices + n_cols - 1) // n_cols, 6)
    indices_to_plot = min(num_indices, 14)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten() if num_indices > 1 else [axes]
    
    for i in range(indices_to_plot):
        ax = axes[i]
        ax.hist(classic_Estimator[:, i], bins=10, alpha=0.6, label='Classic MC',
                color='blue')
        ax.hist(control_Variate_Estimator[:, i], bins=10, alpha=0.6, label='Control Variate',
                color='red')
        ax.set_title(f'Estimator Index {i}', fontsize=10)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for i in range(indices_to_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
