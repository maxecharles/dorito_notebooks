import jax
from jax import random as jr, lax, tree_util as jtu, numpy as jnp
import equinox as eqx
import equinox.internal as eqxi
import lineax as lx
import os
from lineax._custom_types import sentinel
from lineax._misc import inexact_asarray, strip_weak_dtype
from lineax._operator import (
    AbstractLinearOperator,
)
from lineax._solution import RESULTS, Solution
from jaxtyping import ArrayLike, PyTree
from lineax._solve import (
    AutoLinearSolver,
    AbstractLinearSolver,
    IdentityLinearOperator,
    linear_solve_p,
    _linear_solve_impl,
)
from typing import Any
from time import time
from typing import Any, Generic, Optional, TypeVar


os.environ["EQX_ON_ERROR"] = "nan"

print("JAX version:", jax.__version__)
# print("Lineax version:", lx.__version__)


def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[ArrayLike],
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=True),
    *,
    options: Optional[dict[str, Any]] = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> Solution:

    if eqx.is_array(operator):
        raise ValueError(
            "`lineax.linear_solve(operator=...)` should be an "
            "`AbstractLinearOperator`, not a raw JAX array. If you are trying to pass "
            "a matrix then this should be passed as "
            "`lineax.MatrixLinearOperator(matrix)`."
        )
    if options is None:
        options = {}
    vector = jtu.tree_map(inexact_asarray, vector)

    # vector_struct = strip_weak_dtype(jax.eval_shape(lambda: vector))
    # operator_out_structure = strip_weak_dtype(operator.out_structure())
    # `is` to handle tracers
    # if eqx.tree_equal(vector_struct, operator_out_structure) is not True:
    #     raise ValueError(
    #         "Vector and operator structures do not match. Got a vector with structure "
    #         f"{vector_struct} and an operator with out-structure "
    #         f"{operator_out_structure}"
    #     )
    # if isinstance(operator, IdentityLinearOperator):
    #     return Solution(
    #         value=vector,
    #         result=RESULTS.successful,
    #         state=state,
    #         stats={},
    #     )
    if state == sentinel:
        state = solver.init(operator, options)
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = lax.stop_gradient(dynamic_state)
        state = eqx.combine(dynamic_state, static_state)

    state = eqxi.nondifferentiable(state, name="`lineax.linear_solve(..., state=...)`")
    options = eqxi.nondifferentiable(
        options, name="`lineax.linear_solve(..., options=...)`"
    )
    solver = eqxi.nondifferentiable(
        solver, name="`lineax.linear_solve(..., solver=...)`"
    )

    # solution, result, stats = _linear_solve_impl(
    #     operator, state, vector, options, solver, throw, check_closure=False
    # )
    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver, throw
    )
    # return solution
    # TODO: prevent forward-mode autodiff through stats
    stats = eqxi.nondifferentiable_backward(stats)
    return Solution(value=solution, result=result, state=state, stats=stats)


@jax.jit
@jax.grad
def f(matrix, vector):
    print("COMPILING")
    operator = lx.MatrixLinearOperator(matrix)
    return linear_solve(operator, vector, throw=False).value.sum()


# setting up inputs
n = 3
matrix = jr.normal(jr.PRNGKey(0), (n, n))
vector = jr.normal(jr.PRNGKey(1), (n,))
print(type(matrix), type(vector))

# compiling
f(matrix, vector).block_until_ready()

# running five times and tracing with perfetto
with jax.profiler.trace("/tmp/profile-data", create_perfetto_link=True):
    for i in range(5):
        start = time()
        sol = f(matrix, vector).block_until_ready()
        print(sol)
        print(f"Run {i+1} took {time()-start:.4e} seconds")
