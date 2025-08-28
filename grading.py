import jax
import jax.numpy as jnp
import equinox as eqx

jax.print_environment_info()


@eqx.filter_jit
@eqx.filter_grad
def f(x):
    return jnp.sin(x)


x = jnp.array(10.0)

# compiling
f(x).block_until_ready()

with jax.profiler.trace("/tmp/profile-data", create_perfetto_link=True):
    for i in range(5):
        f(x).block_until_ready()
