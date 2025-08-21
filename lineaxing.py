import lineax as lx
import equinox as eqx
import jax
from jax import random as jr
from time import time

print("JAX version:", jax.__version__)
print("Equinox version:", eqx.__version__)
print("Lineax version:", lx.__version__)


@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def f(operator, vector):

    # just do a big computation
    size = 10000
    A = jr.normal(jr.PRNGKey(0), (size, size))
    a = A @ A

    # do the linear solve
    a += lx.linear_solve(operator, vector).value.sum()

    return a


# setting up inputs
matrix = jr.normal(jr.PRNGKey(0), (3, 3))
vector = jr.normal(jr.PRNGKey(1), (3,))
operator = lx.MatrixLinearOperator(matrix)

# compiling
start = time()
f(operator, vector).block_until_ready()
print("Compilation took", time() - start, "seconds")

# running five times and tracing with perfetto
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in range(5):
        start = time()
        f(operator, vector).block_until_ready()
        print(f"Run {i + 1} took {time() - start:.4f} seconds")

print("Done.")
