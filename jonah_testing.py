import jax
print("Using device:", jax.devices()[0])

import jax.numpy as jnp
import jax.random as random
from jax import jit
from tqdm import tqdm
from time import time
import jax.profiler as jp

print('running!')

seed=42
key=random.PRNGKey(seed)
key1,key2=random.split(key)

a=random.uniform(key1, shape=(16384,16384),dtype=jnp.float32)
b=random.uniform(key2, shape=(16384,16384),dtype=jnp.float32)

@jit
def test_fn(a,b):
    """
    Function to test JAX matmul
    """
    return jnp.matmul(a, b)

tic = time()
c = test_fn(a, b)
toc = time()
print(f"Time taken for jit compilation: {toc - tic:.2f} seconds")
 
with jp.trace("/fred/oz440/max/traces/matmul_profile",create_perfetto_trace=True):
    for i in range(100):
        c=test_fn(a,b).block_until_ready()
        print(f"run{i+1}")