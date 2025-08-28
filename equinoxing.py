import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax
import functools as ft
from equinox._errors import _nan_like, _EquinoxRuntimeError
from equinox import _jit

from equinox._errors import _error


# @filter_custom_jvp
def _error(x, pred, index, *, msgs, on_error, stack):
    if on_error == "raise":

        def raises(_index):
            # Sneakily smuggle out the information about the error. Inspired by
            # `sys.last_value`.
            msg = msgs[_index.item()]
            _jit.last_error_info = (msg, stack)
            raise _EquinoxRuntimeError(
                f"{msg}\n\n\n"
                "--------------------\n"
                "An error occurred during the runtime of your JAX program! "
                "Unfortunately you do not appear to be using `equinox.filter_jit` "
                "(perhaps you are using `jax.jit` instead?) and so further information "
                "about the error cannot be displayed. (Probably you are seeing a very "
                "large but uninformative error message right now.) Please wrap your "
                "program with `equinox.filter_jit`.\n"
                "--------------------\n"
            )

        def tpu_msg(_out, _index):
            msg = msgs[_index.item()]
            # `print` doesn't work; nor does `jax.debug.print`.
            # But both `input` and `jax.debug.breakpoint` do. The former allows us to
            # actually display something to the user.
            input(msg + _tpu_msg)
            # We do the tree_map inside the pure_callback, not outside, so that `out`
            # has a data dependency and doesn't get optimised out.
            return jtu.tree_map(_nan_like, _out)

        def handle_error():  # pyright: ignore
            print("a")
            out = jax.pure_callback(raises, struct, index)
            # If we make it this far then we're on the TPU, which squelches runtime
            # errors and returns dummy values instead.
            # Fortunately, we're able to outsmart it!
            return jax.pure_callback(tpu_msg, struct, out, index)

        struct = jax.eval_shape(lambda: x)
        return lax.cond(pred, handle_error, lambda: x)

    elif on_error == "nan":
        return lax.cond(pred, ft.partial(jtu.tree_map, _nan_like), lambda y: y, x)
    else:
        assert False


@jax.jit
def f(x):
    x = _error(x, x < 0, 0, msgs=["x must be >= 0"], on_error="raise", stack=[])
    # x = branched_error_if_impl(x, x < 0, 0, ["x must be >= 0"], on_error="raise")
    x = x - 1
    return x


num = jnp.array(-1)

# compiling
f(num).block_until_ready()

with jax.profiler.trace("/tmp/profile-data", create_perfetto_link=True):
    for i in range(5):
        x = f(num).block_until_ready()
        print(x)
