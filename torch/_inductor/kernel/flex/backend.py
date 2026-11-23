"""What an out-of-tree flex_attention backend receives and returns.

A backend is an :class:`~torch._inductor.choices.InductorChoices` contributor
(``register_inductor_choices``) that overrides
:meth:`~torch._inductor.choices.InductorChoices.lower_flex_attention` and/or
:meth:`~torch._inductor.choices.InductorChoices.lower_flex_attention_backward`.
Inductor calls these first, before it builds anything for its own kernels,
with the call's own arguments (:class:`FlexAttentionCall`,
:class:`FlexAttentionBackwardCall`). A backend either lowers the whole call and
returns every output (:class:`FlexAttentionOutputs`,
:class:`FlexAttentionBackwardOutputs`), or declines: it returns ``None`` or
raises :class:`FlexAttentionDeclined` with a reason, and Inductor lowers the
call itself. Contributors are asked in registration order; the first one that
returns outputs lowers the call.

A backend may also claim ``BACKEND`` names
(:meth:`~torch._inductor.choices.InductorChoices.flex_attention_backends`). A
call made with ``kernel_options={"BACKEND": name}`` is offered only to the
contributors claiming ``name``, carries ``kernel_options["BACKEND_OPTIONS"]``
as ``backend_options``, and fails with the backends' reasons if they decline.
Under ``BACKEND="AUTO"`` every contributor is asked and a decline is silent;
the other built-in names skip contributors.

The contract every backend follows:

- ``logsumexp`` (and ``max_scores``) are float32 ``[B, Hq, M]`` in base 2:
  ``log2(sum_n exp2(score * log2(e)))``. A row that sees no key has
  ``logsumexp = -inf``. The frontend converts them to natural units for the
  user.
- In the backward, ``logsumexp`` is the forward's base-2 one, and
  ``grad_logsumexp`` (when not ``None``) is the gradient of that base-2
  ``logsumexp`` (the frontend's conversion to natural units happens outside
  the call). The backend computes the softmax-backward row terms itself, for
  example ``delta = rowsum(out * grad_out) - grad_logsumexp / ln(2)``, as
  Inductor's own lowering does.
- The BlockMask tables follow ``torch.nn.attention.flex_attention.BlockMask``
  (:class:`FlexBlockMask`). ``full_*`` tables are ``None`` when the mask has
  no full blocks. A call made without a BlockMask carries one block of
  ``_LARGE_SPARSE_BLOCK_SIZE`` covering everything
  (:attr:`FlexBlockMask.is_full_default`).
- ``grad_query``, ``grad_key`` and ``grad_value`` have the shapes and dtypes
  of ``query``, ``key`` and ``value``: a backend that runs a key/value batch
  of 1 against a larger query batch reduces their gradients itself.
- A backend that declines must leave the graph as Inductor would lower it:
  it may realize the inputs it reads (Inductor's lowering realizes them
  too), but must not register buffers or add guards.

A call no backend lowers is lowered by Inductor as before, which still asks
the ``append_flex_attention_choices`` /
``append_flex_attention_backward_choices`` hooks for extra candidates; a call
a backend lowers never reaches them.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE


if TYPE_CHECKING:
    from collections.abc import Mapping

    import sympy

    import torch
    from torch._inductor.ir import TensorBox


__all__ = [
    "FlexAttentionBackwardCall",
    "FlexAttentionBackwardOutputs",
    "FlexAttentionCall",
    "FlexAttentionDeclined",
    "FlexAttentionOutputs",
    "FlexBlockMask",
]


class FlexAttentionDeclined(Exception):
    """Raised by a backend that does not lower a call, with the reason."""


@dataclasses.dataclass(frozen=True)
class FlexBlockMask:
    """A BlockMask, as Inductor lowers it.

    The ``kv_*`` tables list, per query block, the KV blocks it visits (the
    forward walks these); the ``q_*`` tables list, per KV block, the query
    blocks that visit it (the backward's KV-side walk). ``full_*`` tables
    hold the blocks the mask keeps whole, and are ``None`` when the mask has
    none."""

    q_length: sympy.Expr | int
    kv_length: sympy.Expr | int
    kv_num_blocks: TensorBox
    kv_indices: TensorBox
    full_kv_num_blocks: TensorBox | None
    full_kv_indices: TensorBox | None
    q_num_blocks: TensorBox
    q_indices: TensorBox
    full_q_num_blocks: TensorBox | None
    full_q_indices: TensorBox | None
    q_block_size: int
    kv_block_size: int
    mask_mod: torch.fx.GraphModule

    @property
    def is_full_default(self) -> bool:
        """Whether this is the single block the frontend builds for a call
        without a BlockMask (``q_block_size == kv_block_size ==
        _LARGE_SPARSE_BLOCK_SIZE``)."""
        return self.q_block_size == self.kv_block_size == _LARGE_SPARSE_BLOCK_SIZE


@dataclasses.dataclass(frozen=True)
class FlexAttentionCall:
    """A flex_attention forward, as its lowering receives it.

    ``score_mod`` takes ``(score, b, h, m, n, *score_mod_buffers)`` and
    ``block_mask.mask_mod`` takes ``(b, h, m, n, *mask_mod_buffers)``.
    ``kernel_options`` are the call's kernel options without ``BACKEND`` and
    ``BACKEND_OPTIONS``; ``backend`` is the ``BACKEND`` ("AUTO" or a claimed
    name) and ``backend_options`` its ``BACKEND_OPTIONS`` (empty when not
    given). ``output_logsumexp`` and ``output_max`` say which statistics the
    caller reads (outputs it does not read may be returned as ``None``)."""

    query: TensorBox
    key: TensorBox
    value: TensorBox
    score_mod: torch.fx.GraphModule
    block_mask: FlexBlockMask
    scale: float
    score_mod_buffers: tuple[Any, ...]
    mask_mod_buffers: tuple[Any, ...]
    kernel_options: Mapping[str, Any]
    backend: str
    backend_options: Mapping[str, Any]
    output_logsumexp: bool
    output_max: bool


@dataclasses.dataclass(frozen=True)
class FlexAttentionOutputs:
    """Everything a forward returns. ``logsumexp`` and ``max_scores`` may be
    ``None`` when the call does not read them (Inductor then returns empty
    buffers in their place)."""

    out: TensorBox
    logsumexp: TensorBox | None
    max_scores: TensorBox | None


@dataclasses.dataclass(frozen=True)
class FlexAttentionBackwardCall:
    """A flex_attention backward, as its lowering receives it. ``fw_graph``
    is the forward's score_mod and ``joint_graph`` its joint graph:
    ``(score, b, h, m, n, grad_score, *score_mod_buffers)`` to the gradient
    of the score and of each captured buffer. ``kernel_options``,
    ``backend`` and ``backend_options`` are as in :class:`FlexAttentionCall`."""

    query: TensorBox
    key: TensorBox
    value: TensorBox
    out: TensorBox
    logsumexp: TensorBox
    grad_out: TensorBox
    grad_logsumexp: TensorBox | None
    fw_graph: torch.fx.GraphModule
    joint_graph: torch.fx.GraphModule
    block_mask: FlexBlockMask
    scale: float
    score_mod_buffers: tuple[Any, ...]
    mask_mod_buffers: tuple[Any, ...]
    kernel_options: Mapping[str, Any]
    backend: str
    backend_options: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class FlexAttentionBackwardOutputs:
    """Everything a backward returns: the gradients of query, key and value,
    and one entry per captured score_mod buffer (``None`` for a buffer
    without a gradient). A backend that cannot compute a captured buffer's
    gradient declines the call."""

    grad_query: TensorBox
    grad_key: TensorBox
    grad_value: TensorBox
    captured_grads: tuple[TensorBox | None, ...]
