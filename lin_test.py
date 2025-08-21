import jax
from jax import random as jr, lax, tree_util as jtu
import equinox as eqx
import lineax as lx
from lineax._custom_types import sentinel
from lineax._misc import inexact_asarray, strip_weak_dtype
from lineax._operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
)
from lineax._solution import RESULTS, Solution
from jaxtyping import ArrayLike, PyTree
from lineax._solve import AutoLinearSolver, AbstractLinearSolver, linear_solve_p
from typing import Any
import equinox.internal as eqxi


@eqx.filter_jit
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[ArrayLike],
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=True),
    *,
    options: dict[str, Any] | None = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> Solution:
    """
    ...
    """

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
    vector_struct = strip_weak_dtype(jax.eval_shape(lambda: vector))
    operator_out_structure = strip_weak_dtype(operator.out_structure())
    # `is` to handle tracers
    if eqx.tree_equal(vector_struct, operator_out_structure) is not True:
        raise ValueError(
            "Vector and operator structures do not match. Got a vector with structure "
            f"{vector_struct} and an operator with out-structure "
            f"{operator_out_structure}"
        )
    if isinstance(operator, IdentityLinearOperator):
        return Solution(
            value=vector,
            result=RESULTS.successful,
            state=state,
            stats={},
        )
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
    return solver
    # solution, result, stats = eqxi.filter_primitive_bind(
    #     linear_solve_p, operator, state, vector, options, solver, throw
    # )
    # # TODO: prevent forward-mode autodiff through stats
    # stats = eqxi.nondifferentiable_backward(stats)
    # return Solution(value=solution, result=result, state=state, stats=stats)


matrix = jr.normal(jr.PRNGKey(0), (3, 3))
vector = jr.normal(jr.PRNGKey(1), (3,))
operator = lx.MatrixLinearOperator(matrix)

# compile the function
linear_solve(operator, vector)

# running the function multiple times
with jax.profiler.trace("/tmp/jax-trace"):
    for i in range(5):
        linear_solve(operator, vector)
