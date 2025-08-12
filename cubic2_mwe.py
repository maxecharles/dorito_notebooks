import jax
import jax.numpy as jnp
import interpax
from interpax import interp1d
from time import time

print("JAX version:", jax.__version__)
print("Interpax version:", interpax.__version__)

# creating some data for testing
xp = jnp.linspace(0, 2 * jnp.pi, 100)
xq = jnp.linspace(0, 2 * jnp.pi, 10000)
f = lambda x: jnp.sin(x)
fp = f(xp)


# Testing "cubic"
@jax.jit
def cubic_func():
    fq = interp1d(xq, xp, fp, method="cubic")
    return fq


start = time()
cubic_func()  # compiling
print("Compilation time:", time() - start)

# tracing and timing the "cubic" function under jit
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in range(5):
        start = time()
        cubic_func().block_until_ready()  # running under jit
        print(f"Iteration {i + 1} time:", time() - start)


# Testing "cubic2"
@jax.jit
def cubic2_func():
    fq = interp1d(xq, xp, fp, method="cubic2")
    return fq


# compiling "cubic2" function
start = time()
cubic2_func()  # compiling
print("Compilation time:", time() - start)

# tracing and timing the "cubic2" function under jit
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in range(5):
        start = time()
        cubic2_func().block_until_ready()  # running under jit
        print(f"Iteration {i + 1} time:", time() - start)
