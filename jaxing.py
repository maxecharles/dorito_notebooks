import jax
import jax.numpy as jnp
import numpy as np

jax.print_environment_info()


def f_host(x):
    # call a numpy (not jax.numpy) operation:
    return np.sin(x).astype(x.dtype)


@jax.jit
def f(x):
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(f_host, result_shape, x, vmap_method="sequential")


x = jnp.arange(5.0)

# compiling
f(x).block_until_ready()

with jax.profiler.trace("/tmp/profile-data", create_perfetto_link=True):
    for i in range(5):
        f(x).block_until_ready()
