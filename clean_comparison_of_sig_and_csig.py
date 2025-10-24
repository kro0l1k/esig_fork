import jax
import jax.numpy as jnp
import signax

def get_a_path(n, d):
    assert d == 2
    t = jnp.linspace(0, 4 * jnp.pi, n)
    return jnp.concatenate([jnp.sin(t)[:, None], jnp.cos(t)[:, None]], axis=1)

def _tensor(a, b):
    return jnp.einsum("i,j->ij", a, b)

def _tensor3(a, b, c):
    return jnp.einsum("i,j,k->ijk", a, b, c)

# @jax.jit(static_argnames=("order",))
def signature_upto3(path, order: int):
    if order < 1 or order > 3:
        raise ValueError("order must be 1, 2, or 3")

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

    if order == 1:
        return S1
    if order == 2:
        return jnp.concatenate([S1, S2.ravel()])
    return jnp.concatenate([S1, S2.ravel(), S3.ravel()])

# @jax.jit(static_argnames=("order",))
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
    if order < 1 or order > 3:
        raise ValueError("order must be 1, 2, or 3")

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

    if order == 1:
        return C1
    if order == 2:
        return jnp.concatenate([C1, C2.ravel()])
    return jnp.concatenate([C1, C2.ravel(), C3.ravel()])

def main():
    n, d, order = 10, 2, 3
    path = get_a_path(n, d)

    sig_lib = signax.signature(path, order)  # your installed signax uses (path, order)
    sig_manual = signature_upto3(path, order)
    ctrl_manual = controlled_signature_upto3(path, order)

    print("signax vs manual max|.|:", jnp.max(jnp.abs(sig_lib - sig_manual)).item())
    print("controlled vector shape:", ctrl_manual.shape)


    print("signax signature:\n", sig_lib)
    print("manual signature:\n", sig_manual)
    print("manual controlled sig:\n", ctrl_manual)
if __name__ == "__main__":
    main()
