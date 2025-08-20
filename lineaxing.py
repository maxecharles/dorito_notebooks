import lineax as lx
import equinox as eqx
import jax
from jax import numpy as jnp

# flags for debugging
jax.config.update("jax_log_compiles", True)
jax.config.update("jax_explain_cache_misses", True)

print("JAX version:", jax.__version__)
print("Equinox version:", eqx.__version__)
print("Lineax version:", lx.__version__)


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def f(diag, lower_diag, upper_diag, b):
    A = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
    solve = lambda b: lx.linear_solve(A, b, lx.Tridiagonal()).value
    fx = jnp.vectorize(solve, signature="(n)->(n)")(b.T).T
    return fx


# setting up inputs
n = 5
diag = jnp.ones(n)
lower_diag = jnp.zeros(n - 1)
upper_diag = jnp.zeros(n - 1)
b = jnp.linspace(0, 1, n)

# compiling
f(diag, lower_diag, upper_diag, b).block_until_ready()
print("Compilation done.")

# running five times and tracing with perfetto
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in range(5):
        f(diag, lower_diag, upper_diag, b).block_until_ready()
