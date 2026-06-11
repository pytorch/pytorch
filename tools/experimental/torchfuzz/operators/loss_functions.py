# pyre-strict
"""Loss function Operator subclasses.

Covers elementwise losses (mse, huber, smooth_l1, binary_cross_entropy,
binary_cross_entropy_with_logits, soft_margin, hinge_embedding), a
three-input ranking loss (margin_ranking), and classification losses
(nll_loss, cross_entropy).

All operators use ``reduction='none'`` so the output shape is predictable
from the input shape and the fuzzer's top-down graph construction can
compose loss ops as interior nodes at arbitrary depth.

Subclasses that need constrained input ranges (e.g. BCE requires [0, 1])
or +1/-1 target values transform the raw fuzzer-generated tensors inline
in codegen rather than relying on the materializer to produce valid values.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


def _pm1_target_expr(target_name: str) -> str:
    """Emit ``torch.where(target >= 0, +1, -1)`` with ``+-1`` typed to ``target.dtype``.

    Used by ``soft_margin_loss``, ``hinge_embedding_loss``, and
    ``margin_ranking_loss``, which all require a +-1 target.

    The +1/-1 scalars are emitted as 0-dim tensors typed to ``target.dtype`` so the
    wrapped target keeps the input dtype. The naive ``torch.where(cond, 1.0, -1.0)``
    would collapse to the default float dtype (= float32) for any non-default input
    dtype, breaking the spec contract for f16/bf16/f64. ``torch.where`` broadcasts
    0-dim tensors transparently, so this matches the bare-scalar semantics at a
    fraction of the allocation cost of ``torch.ones_like(target)``.
    """
    return (
        f"torch.where({target_name} >= 0, "
        f"torch.tensor(1.0, dtype={target_name}.dtype), "
        f"torch.tensor(-1.0, dtype={target_name}.dtype))"
    )


# ---------------------------------------------------------------------------
# Base class for same-shape float losses (2 or 3 inputs)
# ---------------------------------------------------------------------------


class _SameShapeFloatLossBase(Operator):
    """N same-shape float tensors in, one same-shape float tensor out.

    Subclass hooks:
    - ``_num_inputs`` class attribute (default 2) -- number of input tensors
      requested in ``fuzz_inputs_specs``.
    - ``_transform_inputs(input_names) -> list[str]`` -- per-position expression
      transforms; default returns ``input_names`` unchanged.
    - ``_extra_kwargs() -> dict[str, object]`` -- extra keyword args appended
      to the call (joined into the source via ``f"{k}={v!r}"``); default empty.
    """

    _num_inputs: int = 2

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        return [spec] * self._num_inputs

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        return list(input_names)

    def _extra_kwargs(self) -> dict[str, object]:
        return {}

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        exprs = self._transform_inputs(input_names)
        kwargs: dict[str, object] = {"reduction": "none", **self._extra_kwargs()}
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{output_name} = {self.torch_op_name}({', '.join(exprs)}, {kwargs_str})"


# ---------------------------------------------------------------------------
# Group A: elementwise (2-input) loss subclasses
# ---------------------------------------------------------------------------


class MseLossOperator(_SameShapeFloatLossBase):
    def __init__(self) -> None:
        super().__init__("torch.nn.functional.mse_loss")

    @property
    def torch_op_name(self) -> str:
        return self.name


class HuberLossOperator(_SameShapeFloatLossBase):
    def __init__(self) -> None:
        super().__init__("torch.nn.functional.huber_loss")
        self._delta: float = 0.1

    @property
    def torch_op_name(self) -> str:
        return self.name

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._delta = random.choice([0.1, 0.5, 1.0, 2.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"delta": self._delta}
        return out


class SmoothL1LossOperator(_SameShapeFloatLossBase):
    def __init__(self) -> None:
        super().__init__("torch.nn.functional.smooth_l1_loss")
        self._beta: float = 0.0

    @property
    def torch_op_name(self) -> str:
        return self.name

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # beta=0.0 reduces smooth_l1 to L1 loss; PyTorch requires beta >= 0.
        self._beta = random.choice([0.0, 0.5, 1.0, 2.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"beta": self._beta}
        return out


class BinaryCrossEntropyWithLogitsOperator(_SameShapeFloatLossBase):
    def __init__(self) -> None:
        super().__init__("torch.nn.functional.binary_cross_entropy_with_logits")

    @property
    def torch_op_name(self) -> str:
        return self.name


class BinaryCrossEntropyOperator(_SameShapeFloatLossBase):
    """Wraps both input and target through sigmoid to ensure [0, 1] range."""

    def __init__(self) -> None:
        super().__init__("torch.nn.functional.binary_cross_entropy")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        return [f"torch.sigmoid({n})" for n in input_names]


class SoftMarginLossOperator(_SameShapeFloatLossBase):
    """Wraps target to +1/-1 via :func:`_pm1_target_expr`."""

    def __init__(self) -> None:
        super().__init__("torch.nn.functional.soft_margin_loss")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        input_name, target_name = input_names
        return [input_name, _pm1_target_expr(target_name)]


class HingeEmbeddingLossOperator(_SameShapeFloatLossBase):
    """Wraps target to +1/-1 via :func:`_pm1_target_expr`; randomized margin."""

    def __init__(self) -> None:
        super().__init__("torch.nn.functional.hinge_embedding_loss")
        self._margin: float = 0.0

    @property
    def torch_op_name(self) -> str:
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        input_name, target_name = input_names
        return [input_name, _pm1_target_expr(target_name)]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._margin = random.choice([0.0, 0.5, 1.0, 2.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"margin": self._margin}
        return out


# ---------------------------------------------------------------------------
# Group B: three-input ranking loss (folded into the same base via _num_inputs)
# ---------------------------------------------------------------------------


class MarginRankingLossOperator(_SameShapeFloatLossBase):
    """Three same-shape float inputs (input1, input2, target).

    Target is wrapped to +1/-1 via :func:`_pm1_target_expr`; randomized margin.
    """

    _num_inputs: int = 3

    def __init__(self) -> None:
        super().__init__("torch.nn.functional.margin_ranking_loss")
        self._margin: float = 0.0

    @property
    def torch_op_name(self) -> str:
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        i1, i2, target_name = input_names
        return [i1, i2, _pm1_target_expr(target_name)]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._margin = random.choice([0.0, 0.1, 0.5, 1.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"margin": self._margin}
        return out


# ---------------------------------------------------------------------------
# Group C: classification losses (asymmetric input/output shapes)
# ---------------------------------------------------------------------------


class NllLossOperator(Operator):
    """F.nll_loss(log_softmax(input), target % C, reduction='none').

    Input: (N, C) float -- wrapped through log_softmax to produce valid
    log-probabilities.
    Target: (N,) int64 -- wrapped with .abs() % C to produce valid class
    indices in [0, C).
    Output: (N,) float.

    C (number of classes) is randomly chosen in [2, 10] at
    fuzz_inputs_specs time and stashed for codegen.
    """

    def __init__(self) -> None:
        super().__init__("torch.nn.functional.nll_loss")
        self._num_classes: int | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # Output of F.nll_loss(reduction='none') is always a contiguous (N,) tensor;
        # we can't honor a non-contiguous output stride without an extra
        # as_strided/copy step, so the stride == (1,) filter rejects those specs
        # instead of silently producing a contiguous result.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in FLOAT_DTYPES
            and len(output_spec.size) == 1
            and output_spec.size[0] > 0
            and output_spec.stride == (1,)
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        n = output_spec.size[0]
        self._num_classes = random.randint(2, 10)
        c = self._num_classes
        input_spec = TensorSpec(size=(n, c), stride=(c, 1), dtype=output_spec.dtype)
        target_spec = TensorSpec(size=(n,), stride=(1,), dtype=torch.int64)
        return [input_spec, target_spec]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        c = self._num_classes
        out = (
            f"{output_name} = torch.nn.functional.nll_loss("
            f"torch.log_softmax({input_names[0]}, dim=1), "
            f"{input_names[1]}.abs() % {c}, reduction='none')"
        )
        self._num_classes = None
        return out


class CrossEntropyOperator(Operator):
    """F.cross_entropy(input, target % C, reduction='none').

    Input: (N, C) float -- raw logits (cross_entropy applies log_softmax
    internally).
    Target: (N,) int64 -- wrapped with .abs() % C to produce valid class
    indices in [0, C).
    Output: (N,) float.

    C (number of classes) is randomly chosen in [2, 10] at
    fuzz_inputs_specs time and stashed for codegen.
    """

    def __init__(self) -> None:
        super().__init__("torch.nn.functional.cross_entropy")
        self._num_classes: int | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # Output of F.cross_entropy(reduction='none') is always a contiguous (N,)
        # tensor; we can't honor a non-contiguous output stride without an extra
        # as_strided/copy step, so the stride == (1,) filter rejects those specs
        # instead of silently producing a contiguous result.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in FLOAT_DTYPES
            and len(output_spec.size) == 1
            and output_spec.size[0] > 0
            and output_spec.stride == (1,)
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        n = output_spec.size[0]
        self._num_classes = random.randint(2, 10)
        c = self._num_classes
        input_spec = TensorSpec(size=(n, c), stride=(c, 1), dtype=output_spec.dtype)
        target_spec = TensorSpec(size=(n,), stride=(1,), dtype=torch.int64)
        return [input_spec, target_spec]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        c = self._num_classes
        out = (
            f"{output_name} = torch.nn.functional.cross_entropy("
            f"{input_names[0]}, "
            f"{input_names[1]}.abs() % {c}, reduction='none')"
        )
        self._num_classes = None
        return out
