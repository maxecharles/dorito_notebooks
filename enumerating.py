import jax
import jax.numpy as jnp
from jax import tree_util as jtu, Array
from equinox._unvmap import unvmap_any, unvmap_max
from equinox._errors import EquinoxTracetimeError, _nan_like
from equinox._config import EQX_ON_ERROR
import warnings


def branched_error_if_impl(
    x,
    pred,
    index,
    msgs,
    *,
    on_error,
):
    if on_error == "default":
        on_error = EQX_ON_ERROR
    elif on_error not in ("raise", "breakpoint", "nan"):
        raise RuntimeError("Unrecognised value for `on_error`.")
    with jax.ensure_compile_time_eval():
        # This carefully does not perform any JAX operations if `pred` and `index` are
        # a bool and an int.
        # This ensures we can use `error_if` before init_google.
        print("a")
        if not isinstance(pred, bool):
            pred = unvmap_any(pred)
            print("b")

        if not isinstance(index, int):
            index = unvmap_max(index)
            print("c")

        if not isinstance(pred, jax.core.Tracer):
            print("d")
            if isinstance(pred, Array):
                pred = pred.item()
                print("e")
            assert type(pred) is bool
            if pred:
                print("f")
                if not isinstance(index, jax.core.Tracer):
                    print("g")
                    if isinstance(index, Array):
                        index = index.item()
                    assert type(index) is int
                    if on_error == "raise":
                        raise EquinoxTracetimeError(msgs[index])
                    elif on_error == "breakpoint":
                        print(msgs[index])
                        breakpoint()
                    elif on_error == "nan":
                        warnings.warn(
                            "Resolving error at trace time (because the predicate is "
                            "statically resolvable), by substituting NaNs (because "
                            "`on_error='nan'`)."
                        )
                        return jtu.tree_map(_nan_like, x)
                    else:
                        assert False
                # else defer error to runtime, when the index is known.
        else:
            print("h")
            return x


@jax.jit
def f(x):
    x = branched_error_if_impl(x, x < 0, 0, ["x must be >= 0"], on_error="raise")
    print(x)
    x = x - 1
    return x


num = jnp.array(10)

# compiling
f(num).block_until_ready()

with jax.profiler.trace("/tmp/profile-data", create_perfetto_link=True):
    for i in range(5):
        f(num).block_until_ready()
