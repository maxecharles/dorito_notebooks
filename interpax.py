# %%
import interpax as ipx
import jax
import equinox as eqx
import jax.numpy as jnp
import dLux.utils as dlu
from time import time
import lineax as lx

# %%
n = 500
coords = dlu.nd_coords(n)
knots = jnp.linspace(-1, 1, n)

# %%


@eqx.filter_jit
def _cubic1(x, f, axis):
    dx = jnp.diff(x)
    df = jnp.diff(f, axis=axis)
    dxi = jnp.where(dx == 0, 0, 1 / dx)
    if df.ndim > dxi.ndim:
        dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
        dxi = jnp.moveaxis(dxi, 0, axis)
    df = dxi * df
    fx = jnp.concatenate(
        [
            jnp.take(df, jnp.array([0]), axis, mode="wrap"),
            1
            / 2
            * (
                jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                + jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
            ),
            jnp.take(df, jnp.array([-1]), axis, mode="wrap"),
        ],
        axis=axis,
    )
    return fx


# compiling
_cubic1(coords, knots, axis=0)

# profiling
# tracing and timing the "cubic2" function under jit
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in range(5):
        start = time()
        _cubic1(coords, knots, axis=0).block_until_ready()  # running under jit
        print(f"Iteration {i + 1} time:", time() - start)


# # %%
# @eqx.filter_jit
# def _cubic2(x, f, axis, bc, dtype):
#     f = f.astype(dtype)
#     f = jnp.moveaxis(f, axis, 0)
#     dx = jnp.diff(x)
#     df = jnp.diff(f, axis=0)
#     dxr = dx.reshape([dx.shape[0]] + [1] * (f.ndim - 1))
#     dxi = jnp.where(dxr == 0, 0, 1 / jnp.where(dxr == 0, 1, dxr))
#     df = dxi * df
#     n = len(f)

#     # If bc is 'not-a-knot' this change is just a convention.
#     # If bc is 'periodic' then we already checked that y[0] == y[-1],
#     # and the spline is just a constant, we handle this case in the
#     # same way by setting the first derivatives to slope, which is 0.
#     if n == 2:
#         if bc[0] in ["not-a-knot", "periodic"]:
#             bc[0] = (1, df[0])
#         if bc[1] in ["not-a-knot", "periodic"]:
#             bc[1] = (1, df[0])

#     # This is a special case, when both conditions are 'not-a-knot'
#     # and n == 3. In this case 'not-a-knot' can't be handled regularly
#     # as the both conditions are identical. We handle this case by
#     # constructing a parabola passing through given points.
#     if n == 3 and bc[0] == "not-a-knot" and bc[1] == "not-a-knot":
#         A = jnp.zeros((3, 3))  # This is a standard matrix.
#         b = jnp.empty((3,) + f.shape[1:], dtype=dtype)

#         A = A.at[0, 0].set(1)
#         A = A.at[0, 1].set(1)
#         A = A.at[1, 0].set(dx[1])
#         A = A.at[1, 1].set(2 * (dx[0] + dx[1]))
#         A = A.at[1, 2].set(dx[0])
#         A = A.at[2, 1].set(1)
#         A = A.at[2, 2].set(1)

#         b = b.at[0].set(2 * df[0])
#         b = b.at[1].set(3 * (dxr[0] * df[1] + dxr[1] * df[0]))
#         b = b.at[2].set(2 * df[1])

#         solve = lambda b: jnp.linalg.solve(A, b)
#         fx = jnp.vectorize(solve, signature="(n)->(n)")(b.T).T
#         fx = jnp.moveaxis(fx, 0, axis)

#     else:
#         # Find derivative values at each x[i] by solving a tridiagonal
#         # system.
#         diag = jnp.zeros(n, dtype=x.dtype)
#         diag = diag.at[1:-1].set(2 * (dx[:-1] + dx[1:]))
#         upper_diag = jnp.zeros(n - 1, dtype=x.dtype)
#         upper_diag = upper_diag.at[1:].set(dx[:-1])
#         lower_diag = jnp.zeros(n - 1, dtype=x.dtype)
#         lower_diag = lower_diag.at[:-1].set(dx[1:])
#         b = jnp.zeros((n,) + f.shape[1:], dtype=dtype)
#         b = b.at[1:-1].set(3 * (dxr[1:] * df[:-1] + dxr[:-1] * df[1:]))

#         bc_start, bc_end = bc

#         if bc_start == "not-a-knot":
#             d = x[2] - x[0]
#             diag = diag.at[0].set(dx[1])
#             upper_diag = upper_diag.at[0].set(d)
#             b = b.at[0].set(
#                 ((dxr[0] + 2 * d) * dxr[1] * df[0] + dxr[0] ** 2 * df[1]) / d
#             )
#         else:

#             def bc_start0(diag, upper_diag, b):
#                 return diag, upper_diag, b

#             def bc_start1(diag, upper_diag, b):
#                 diag = diag.at[0].set(1)
#                 upper_diag = upper_diag.at[0].set(0)
#                 b = b.at[0].set(bc_start[1])
#                 return diag, upper_diag, b

#             def bc_start2(diag, upper_diag, b):
#                 diag = diag.at[0].set(2 * dx[0])
#                 upper_diag = upper_diag.at[0].set(dx[0])
#                 b = b.at[0].set(-0.5 * bc_start[1] * dx[0] ** 2 + 3 * (f[1] - f[0]))
#                 return diag, upper_diag, b

#             diag, upper_diag, b = jax.lax.cond(
#                 bc_start[0] == 1, bc_start1, bc_start0, diag, upper_diag, b
#             )
#             diag, upper_diag, b = jax.lax.cond(
#                 bc_start[0] == 2, bc_start2, bc_start0, diag, upper_diag, b
#             )

#         if bc_end == "not-a-knot":
#             d = x[-1] - x[-3]
#             diag = diag.at[-1].set(dx[-2])
#             lower_diag = lower_diag.at[-1].set(d)
#             b = b.at[-1].set(
#                 (dxr[-1] ** 2 * df[-2] + (2 * d + dxr[-1]) * dxr[-2] * df[-1]) / d
#             )
#         else:

#             def bc_end0(diag, lower_diag, b):
#                 return diag, lower_diag, b

#             def bc_end1(diag, lower_diag, b):
#                 diag = diag.at[-1].set(1)
#                 lower_diag = lower_diag.at[-1].set(0)
#                 b = b.at[-1].set(bc_end[1])
#                 return diag, lower_diag, b

#             def bc_end2(diag, lower_diag, b):
#                 diag = diag.at[-1].set(2 * dx[-1])
#                 lower_diag = lower_diag.at[-1].set(dx[-1])
#                 b = b.at[-1].set(0.5 * bc_end[1] * dx[-1] ** 2 + 3 * (f[-1] - f[-2]))
#                 return diag, lower_diag, b

#             diag, lower_diag, b = jax.lax.cond(
#                 bc_end[0] == 1, bc_end1, bc_end0, diag, lower_diag, b
#             )
#             diag, lower_diag, b = jax.lax.cond(
#                 bc_end[0] == 2, bc_end2, bc_end0, diag, lower_diag, b
#             )

#         # this is needed to avoid singular matrix when there are duplicate x coords
#         mask = diag == 0
#         diag = jnp.where(mask, 1, diag)
#         lower_diag = jnp.where(mask[1:], 0, lower_diag)
#         upper_diag = jnp.where(mask[:-1], 0, upper_diag)
#         b = jnp.where(mask, 0, b.T).T

#         # see https://github.com/patrick-kidger/lineax/issues/148
#         dtype = jnp.result_type(diag, lower_diag, upper_diag, b)
#         diag = diag.astype(dtype)
#         lower_diag = lower_diag.astype(dtype)
#         upper_diag = upper_diag.astype(dtype)
#         b = b.astype(dtype)

#         A = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)

#         solve = lambda b: lx.linear_solve(A, b, lx.Tridiagonal()).value
#         fx = jnp.vectorize(solve, signature="(n)->(n)")(b.T).T
#         fx = jnp.moveaxis(fx, 0, axis)
#     return fx.astype(f.dtype)


# # compiling
# _cubic2(coords, knots, axis=0, bc="not-a-knot", dtype=jnp.float32)

# # profiling
# # tracing and timing the "cubic2" function under jit
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     for i in range(5):
#         start = time()
#         _cubic2(
#             coords, knots, axis=0, bc="not-a-knot", dtype=jnp.float32
#         ).block_until_ready()  # running under jit
#         print(f"Iteration {i + 1} time:", time() - start)

# %%
