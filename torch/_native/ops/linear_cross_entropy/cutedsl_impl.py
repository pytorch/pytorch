"""CuTeDSL override registrations for the chunked ``linear_cross_entropy`` ops.

`torch_nn::_linear_cross_entropy_batch_chunked` and its
`..._no_reduction` sibling each put a whole fused loss behind one dispatcher
symbol: an ``(N, F) x (F, C)`` logits matmul, a row-shifted softmax, the
cross-entropy reduction, and -- on the scalar-reduction op when gradients are
requested -- three gradient matmuls, all chunked over the batch. Per chunk the
eager loop walks the ``(B, C)`` logits buffer roughly seven times (matmul write,
row max, subtract, gather, ``exp_``, row sum, scale) before the gradient matmuls
read it again. Collapsing those passes is what a DSL kernel is for.

``_OVERRIDES`` is the single list of what this module registers; the
registration loop and the tests both read it, so adding an override is one row.
"""

import importlib

import torch

from ... import cutedsl_utils as cu


def _batch_chunked_cond(*args: object, **kwargs: object) -> bool:
    # Where a kernel's eligibility gate goes: it returns True only for the
    # shapes and dtypes that kernel implements, and the registry's router falls
    # back to the op for everything else.
    return True


def _batch_chunked_impl(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    allow_retain_graph: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
    compute_linear_bias_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Installing at a backend key means the op's body never runs, so its checks
    # have to be run here. ``allow_retain_graph`` is a backward-only flag that
    # neither the checks nor the accumulator take.
    #
    # A kernel replacing the accumulator call below also loses the checks
    # *inside* it and must re-enforce those: the unresolved
    # ``acc_policy`` / ``acc_dtype`` check, and ``_ChunkContext.build``'s
    # ``linear_bias`` shape and ``acc_dtype`` compatibility checks.
    from torch.nn.modules.linear_cross_entropy import (
        _check_batch_chunked_grad_flags,
        _linear_cross_entropy_batch_chunked_accumulator,
    )

    _check_batch_chunked_grad_flags(
        input,
        linear_weight,
        target,
        linear_bias,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    )
    return _linear_cross_entropy_batch_chunked_accumulator(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    )


def _no_reduction_cond(*args: object, **kwargs: object) -> bool:
    # As above: where this op's kernel gate goes.
    return True


def _no_reduction_impl(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    ignore_index: int,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
) -> torch.Tensor:
    # Mirrors the op's body: the accumulator's loss-only branch, selected by
    # reduction='none' with nothing precomputed. No checks to call -- this op
    # takes no compute_*_grad flags, and its probability-target guard lives in
    # its backward, which is registered above this dispatch key and so is not
    # displaced by the override.
    #
    # A kernel replacing this call inherits the same obligation as the scalar
    # op's impl: re-enforce the checks inside the accumulator.
    from torch.nn.modules.linear_cross_entropy import (
        _linear_cross_entropy_batch_chunked_accumulator,
    )

    return _linear_cross_entropy_batch_chunked_accumulator(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        "none",
        ignore_index,
        0.0,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad=False,
        compute_linear_weight_grad=False,
        compute_linear_bias_grad=False,
    )[0]


# (op_symbol, cond, impl) for every override this module installs on the
# `torch_nn` namespace. Single source of truth: the registration loop below and
# test/python_native/test_linear_cross_entropy_override.py both read it.
# Declaration read by both the registrar below and the drift-guard test in
# test/python_native/test_override_declarations.py. `aten` ops exist by
# construction, so a bad symbol there dies on any `import torch`; a
# `torch_nn` op exists only once `_DEFINING_MODULE` has executed, which makes
# the binding a runtime property whose failures surface only where the DSL is
# installed. The test resolves these symbols with neither a GPU nor the DSL,
# so drift cannot ship.
_NAMESPACE = "torch_nn"
_DEFINING_MODULE = "torch.nn.modules.linear_cross_entropy"
_OVERRIDES = (
    ("_linear_cross_entropy_batch_chunked", _batch_chunked_cond, _batch_chunked_impl),
    (
        "_linear_cross_entropy_batch_chunked_no_reduction",
        _no_reduction_cond,
        _no_reduction_impl,
    ),
)


def register_linear_cross_entropy_overrides() -> None:
    # Bail out before the import below when the DSL is unavailable or disabled;
    # cu.register_op_override would drop the registration anyway, and the
    # import is not free. Don't gate on torch.cuda.is_available() here -- it
    # calls cuInit and poisons fork.
    if not cu.runtime_available() or cu.check_native_jit_disabled():
        return

    # This import is what defines the ops named in `_OVERRIDES`. torch.nn cannot pull
    # the module in from torch/nn/modules/__init__.py (torch.library does not
    # exist that early in `import torch`), so it is imported lazily on first
    # use -- meaning the ops are absent at registration time unless something
    # asks for them first. Overrides are installed against ops that already
    # exist in the dispatcher, hence the import here.
    importlib.import_module(_DEFINING_MODULE)

    for op_symbol, cond, impl in _OVERRIDES:
        cu.register_op_override(_NAMESPACE, op_symbol, "CUDA", cond=cond, impl=impl)
