import jax
from jax import numpy as jnp, random as jr
import equinox as eqx
from time import time
import lineax as lx


@eqx.filter_jit
def f(diag, lower_diag, upper_diag, b):
    print("Compiling")
    A = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
    solve = lambda b: lx.linear_solve(A, b, lx.Tridiagonal()).value
    fx = jnp.vectorize(solve, signature="(n)->(n)")(b.T).T
    return fx


# setting up inputs
n = 10000
diag = jnp.ones(n)
lower_diag = jnp.zeros(n - 1)
upper_diag = jnp.zeros(n - 1)
b = jnp.linspace(0, 1, n)

# compiling
start = time()
f(diag, lower_diag, upper_diag, b).block_until_ready()
print("Compilation took", time() - start, "seconds")

with jax.profiler.trace("/tmp/profile-data"):
    for i in range(5):
        start = time()
        f(diag, lower_diag, upper_diag, b).block_until_ready()
        print(f"Run {i + 1} took {time() - start:.4f} seconds")
