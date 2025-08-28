import lineax as lx
import jax
from jax import random as jr

print("JAX version:", jax.__version__)
print("Lineax version:", lx.__version__)


@jax.jit
@jax.grad
def f(matrix, vector):
    print("COMPILING")
    operator = lx.MatrixLinearOperator(matrix)
    return lx.linear_solve(operator, vector, throw=False).value.sum()


# setting up inputs
matrix = jr.normal(jr.PRNGKey(0), (3, 3))
vector = jr.normal(jr.PRNGKey(1), (3,))
print(type(matrix), type(vector))

# compiling
f(matrix, vector).block_until_ready()

# running five times and tracing with perfetto
with jax.profiler.trace("/tmp/profile-data", create_perfetto_link=True):
    for i in range(5):
        f(matrix, vector).block_until_ready()
