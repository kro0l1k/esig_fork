import jax
import jax.numpy as jnp
import signax


def get_a_path(n, d):
    assert d == 2
    t = jnp.linspace(0, 4 * jnp.pi, n)
    path1 = jnp.sin(t).reshape(-1, 1)
    path2 = jnp.cos(t).reshape(-1, 1)
    return jnp.concatenate([path1, path2], axis=1)


def _tensor(a, b):
    return jnp.einsum("i,j->ij", a, b)


def _tensor3(a, b, c):
    return jnp.einsum("i,j,k->ijk", a, b, c)


# @jax.jit
def get_manual_signature(path, order: int):
    """
    Signature up to level `order` ∈ {1,2,3} for a piecewise-linear path using Chen's identity.
    Flattens level-wise in C order to match signax.
    """
    if order < 1 or order > 3:
        raise ValueError("order must be 1, 2, or 3")

    inc = jnp.diff(path, axis=0)  # (n-1, d)
    d = inc.shape[1]

    S1 = jnp.zeros((d,))
    S2 = jnp.zeros((d, d))
    S3 = jnp.zeros((d, d, d))

    def body(i, state):
        S1, S2, S3 = state
        a = inc[i]                          # segment increment
        a2 = 0.5 * _tensor(a, a)            # 1/2 a⊗a
        a3 = (1.0 / 6.0) * _tensor3(a, a, a)  # 1/6 a⊗a⊗a

        # Chen product: Sig ⊗ exp(a)
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


def main():
    n, d, order = 1550, 2, int(3) 
    path = get_a_path(n, d)

    sig_lib = signax.signature(path, depth=order)  # signax>=0.2 uses depth=
    sig_manual = get_manual_signature(path, order)

    print("signax shape:", sig_lib.shape)
    print("manual shape:", sig_manual.shape)
    print("max abs diff:", jnp.max(jnp.abs(sig_lib - sig_manual)).item())


if __name__ == "__main__":
    main()
