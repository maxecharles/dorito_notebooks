print("Running test script...")

import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

print("Using device:", jax.devices()[0])
# print("Default Backend:", jax.default_backend())


key = jax.random.PRNGKey(0)
size = 1000
x = jax.random.normal(key, (size, size), dtype=jnp.float64)
x = jax.device_put(x)

@jax.jit
def matmul(x):
    return jnp.dot(x, x)

# Warmup
matmul(x).block_until_ready()

for _ in range(10):
    start = time.time()
    x = matmul(x).block_until_ready()
    end = time.time()
    print("Total time:", end - start, "seconds")

