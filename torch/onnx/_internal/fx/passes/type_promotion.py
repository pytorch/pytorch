# Owner(s): ["module: onnx"]
from __future__ import annotations

import abc

import dataclasses
import inspect
import logging
from types import ModuleType

from typing import Any, Callable, Mapping, Optional, Sequence, Set

import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback

from torch import _prims_common, _refs

from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    wrappers as _prims_common_wrappers,
)
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor

# Imported to resolve beartype issue when type checking node.Argument.
from torch.fx.node import Node  # noqa: F401

from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree

logger = logging.getLogger(__name__)

# TODO(bowbao): move to type utils.
_SCALAR_TYPE_TENSOR_DTYPE_MAP: Mapping[type, torch.dtype] = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex32,
}


def _try_getclosurevars(func):
    try:
        return inspect.getclosurevars(func)
    except TypeError as e:
        return None


@dataclasses.dataclass
class TypePromotionSnapshot:
    """Type promotion snapshot for a fx node and its inputs.

    Contains the promoted dtype for args and kwargs that needs promoting.
    Contains the expected node output dtype.
    """

    args_dtypes: Mapping[int, torch.dtype]
    """Mapping from arg position to dtype to promote to."""

    kwargs_dtypes: Mapping[str, torch.dtype]
    """Mapping from kwarg name to dtype to promote to."""

    out_dtype: torch.dtype
    """Expected output dtype of the node."""


@_beartype.beartype
def _fake_tensor_from_node_val(node: torch.fx.Node) -> fake_tensor.FakeTensor:
    """Syntactic sugar for retrieving fake tensor from node.meta['val']."""
    val = node.meta.get("val", None)
    if not isinstance(val, fake_tensor.FakeTensor):
        raise RuntimeError(
            f"Cannot retrieve fake tensor from node {node}. Got type({type(val)}) instead."
        )
    return val


class TypePromotionRule(abc.ABC):
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""

    def __init__(self, namespace: str, op_name: str):
        self.namespace = namespace
        self.op_name = op_name

    # Make this abstract as well because subclass needs to override __eq__().
    # A class that overrides __eq__() and does not define __hash__() will have its __hash__() implicitly set to None.
    # Ref: https://docs.python.org/3/reference/datamodel.html#object.__hash__
    @abc.abstractmethod
    def __hash__(self) -> int:
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...

    def is_valid(self) -> bool:
        """Check if the rule is valid."""
        # This always returns a module. If the module does not exist it will be created.
        module = getattr(torch.ops, self.namespace)
        py_op = getattr(module, self.op_name, None)
        if py_op is None:
            logger.warning(
                "Cannot find op: %s in module: %s", self.op_name, self.namespace
            )
            return False
        if not isinstance(py_op, torch._ops.OpOverloadPacket):
            logger.warning(
                "Op: torch.ops.%s.%s is not an OpOverloadPacket, got: %s",
                self.namespace,
                self.op_name,
                type(py_op),
            )
            return False

        return True

    @abc.abstractmethod
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        """Preview type promotion results for provided set of args and kwargs.

        Returns a TypePromotionSnapshot object that contains the promoted dtypes for
        the arguments and the expected output dtype.
        """
        ...


class ElementwiseTypePromotionRule(TypePromotionRule):
    """Defines how to perform elementwise type promotion for 'torch.ops.{namespace}.{op_name}'."""

    def __init__(
        self,
        namespace: str,
        op_name: str,
        promote_args_positions: Sequence[int],
        promote_kwargs_names: Sequence[str],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ):
        """Constructs a TypePromotionRule for elementwise operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.add'.
            op_name: Name of the op. E.g. 'add' in 'torch.ops.aten.add'.
            promote_args_positions: Positions of args to promote.
            promote_kwargs_names: Names of kwargs to promote.
            promotion_kind: Type promotion kind. Refer to [_prims_common.elementwise_dtypes](https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promote_args_positions = promote_args_positions
        self.promote_kwargs_names = promote_kwargs_names
        self.promotion_kind = promotion_kind

    def __repr__(self):
        return (
            f"ElementwiseTypePromotionRule('{self.namespace}', '{self.op_name}', "
            f"{self.promote_args_positions}, {self.promote_kwargs_names}, {self.promotion_kind})"
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == __value.namespace
            and self.op_name == __value.op_name
            and self.promote_args_positions == __value.promote_args_positions
            and self.promote_kwargs_names == __value.promote_kwargs_names
            and self.promotion_kind == __value.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        candidate_args = {
            i: args[i]
            for i in self.promote_args_positions
            if i < len(args) and args[i] is not None
        }
        candidate_kwargs = {
            name: kwargs[name]
            for name in self.promote_kwargs_names
            if name in kwargs and kwargs[name] is not None
        }

        computed_dtype, result_dtype = _prims_common.elementwise_dtypes(
            *_pytree.tree_flatten(candidate_args)[0],
            *_pytree.tree_flatten(candidate_kwargs)[0],
            type_promotion_kind=self.promotion_kind,
        )

        return TypePromotionSnapshot(
            {i: computed_dtype for i in candidate_args.keys()},
            {name: computed_dtype for name in candidate_kwargs.keys()},
            result_dtype,
        )


class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    """Reference type promotion rule from torch._refs.div.

    Rule depends on the value of the `rounding_mode` argument.
    """

    def __init__(self):
        super().__init__(
            "aten",
            "div",
            promote_args_positions=(0, 1),
            promote_kwargs_names=(),
            promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        rounding_mode = kwargs.get("rounding_mode", None)
        if rounding_mode is None:
            # true_divide
            self.promotion_kind = (
                _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            )
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "trunc":
            # trunc_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "floor":
            # floor_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        raise ValueError(f"Unknown rounding_mode: {rounding_mode}")


class ReductionTypePromotionRule(TypePromotionRule):
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND,
    ):
        """Constructs a TypePromotionRule for reduction operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.sum'.
            op_name: Name of the op. E.g. 'sum' in 'torch.ops.aten.sum'.
            promotion_kind: Type promotion kind. Refer to [_prims_common.reduction_dtypes]((https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py)) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promotion_kind = promotion_kind

    def __repr__(self):
        return f"ReductionTypePromotionRule('{self.namespace}', '{self.op_name}', {self.promotion_kind})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == __value.namespace
            and self.op_name == __value.op_name
            and self.promotion_kind == __value.promotion_kind
        )

    def __hash__(self) -> int:
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        assert len(args) >= 1 and isinstance(
            arg := args[0], torch.Tensor
        ), f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        dtype: Optional[torch.dtype] = kwargs.get("dtype", None)

        computation_dtype, result_dtype = _prims_common.reduction_dtypes(
            arg, self.promotion_kind, dtype
        )
        if result_dtype is None:
            # Inspecting code, this can only happen when `promotion_kind` is `KEEP_PROMOTED_TYPE`.
            # Hence set same as computation_dtype.
            result_dtype = computation_dtype

        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.all or torch.ops.aten.any.

    This is a special case where computation dtype is always torch.bool.
    The result dtype is always uint8 if `dtype` kwarg is uint8, otherwise torch.bool.
    """

    def __init__(self, op_name: str):
        super().__init__(
            "aten",
            op_name,
            _prims_common.REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        assert len(args) >= 1 and isinstance(
            arg := args[0], torch.Tensor
        ), f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        computation_dtype = torch.bool
        # Preserves uint8 -- probably a legacy mask thing
        result_dtype = torch.uint8 if arg.dtype == torch.uint8 else torch.bool
        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class SumLikeReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.sum.

    This is a special case where computation dtype is always torch.int64 for integral arg,
    unless overridden by `dtype` kwarg.
    """

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        assert len(args) >= 1 and isinstance(
            arg := args[0], torch.Tensor
        ), f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
        dtype: Optional[torch.dtype] = kwargs.get("dtype", None)
        # The below logic is copied from `torch/_refs/__init__.py` reduction ops impl.
        if dtype is None:
            if _prims_common.is_boolean_dtype(
                arg.dtype
            ) or _prims_common.is_integer_dtype(arg.dtype):
                dtype = torch.int64
            else:
                dtype = arg.dtype
        return super().preview_type_promotion(args, {"dtype": dtype})


# NOTE: [Update type promotion rule]
# BELOW TABLE IS GENERATED FROM `TypePromotionRuleSetGenerator.generate_from_torch_refs`.
# DO NOT EDIT MANUALLY !!!
# For missing rules or discrepancies, please
# 1. Run `pytest test/onnx/test_fx_type_promotion.py` to validate if the generated rule set is current.
#    If it is not, update with new generated set.
# 2. If discrepancies still exist, consider debugging torch._refs or report a bug.
# 3. If rules are still missing, add them to `_EXTRA_TYPE_PROMOTION_RULE_SET` or report a bug.
# Check `TypePromotionRule` class for how each rule is defined and used.
_GENERATED_ATEN_TYPE_PROMOTION_RULE_SET = {
    ElementwiseTypePromotionRule(
        "aten", "abs", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "abs_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addr", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cat", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fill", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmax", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmin", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gelu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "glu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "hardtanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "huber_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "isfinite", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isnan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isneginf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isposinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isreal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "l1_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "le", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "le_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "leaky_relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logsumexp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "maximum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "minimum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mse_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nll_loss", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "pdist", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "poisson_nll_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "pow", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "pow_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "prelu", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "round", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "signbit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "smooth_l1_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "softplus", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "square", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "square_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "where", [1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
}

# Manually curated extra type promotion rules. Please see NOTE [Update type promotion rule]
# before adding new rules.
_EXTRA_TYPE_PROMOTION_RULE_SET = {
    # torch._refs skips type promotion decoration for `clamp_min` and `clamp_max` since
    # the call is routed to the decorated `aten.clamp` op.
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_max",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_min",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # torch.ops.aten.div.Tensor_mode applies different type promotion rules
    # depending on the value of the `mode` argument.
    DivElementwiseTypePromotionRule(),
    # Manually curating reduction ops since the logic is written inside the op reference
    # implementation.
    AllOrAnyReductionTypePromotionRule("all"),
    AllOrAnyReductionTypePromotionRule("any"),
    ReductionTypePromotionRule(
        "aten",
        "amax",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    ReductionTypePromotionRule(
        "aten",
        "amin",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    # torch.ops.aten.mean is a special case that does not need type promotion.
    ReductionTypePromotionRule(
        "aten",
        "std",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "std_mean",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "var",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumprod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumsum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "prod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "sum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
}


class ElementwiseTypePromotionRuleSetGenerator:
    """Hackly distilling info from reference ops decorated with elementwise type promotion rule.

    The goal is to retrieve the decorator

    ```python
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
    ```

    from the reference ops. It provides info as for which arguments are promoted
    and what kind of promotion is applied.
    """

    @classmethod
    def generate_from_torch_refs(cls) -> Set[ElementwiseTypePromotionRule]:
        """Parse type promotion rules from reference ops under torch._C._refs."""
        rule_set = set()
        rule_set.update(cls._parse_torch_refs(_refs))
        rule_set.update(cls._parse_torch_refs(_nn_refs))
        rule_set.update(cls._parse_torch_refs(_linalg_refs))
        rule_set.update(cls._parse_torch_refs(_special_refs))
        rule_set.update(cls._parse_torch_refs(_functional_refs))
        return rule_set

    @classmethod
    def _parse_torch_refs(
        cls, ref_module: ModuleType
    ) -> Set[ElementwiseTypePromotionRule]:
        logger.info("Processing module: %s", ref_module.__name__)
        rule_set = set()
        for name in ref_module.__all__:
            decorated_op = getattr(ref_module, name)
            rule = cls._parse_type_promotion_rule_from_refs_op(decorated_op)
            if rule is not None and rule.is_valid():
                rule_set.add(rule)

        return rule_set

    @classmethod
    def _parse_type_promotion_rule_from_refs_op(
        cls,
        decorated_op: Callable,
    ) -> Optional[ElementwiseTypePromotionRule]:
        """Retrieve and parse type promotion decorator from op under torch._refs."""
        fn = decorated_op
        type_promo_wrapper = None
        while fn_closure_vars := _try_getclosurevars(fn):
            if "fn" not in fn_closure_vars.nonlocals:
                break
            if "self" in fn_closure_vars.nonlocals and isinstance(
                fn_closure_vars.nonlocals["self"],
                _prims_common_wrappers.elementwise_type_promotion_wrapper,
            ):
                type_promo_wrapper = fn_closure_vars.nonlocals["self"]
                break
            fn = fn_closure_vars.nonlocals["fn"]

        if type_promo_wrapper is not None:
            signature = inspect.signature(decorated_op)

            pos = 0
            promote_args_positions = []
            promote_kwargs_names = []

            if type_promo_wrapper.type_promoting_arg_names is not None:
                for name, param in signature.parameters.items():
                    if name in type_promo_wrapper.type_promoting_arg_names:
                        if param.kind in (
                            param.POSITIONAL_OR_KEYWORD,
                            param.POSITIONAL_ONLY,
                        ):
                            promote_args_positions.append(pos)
                        elif param.kind == param.KEYWORD_ONLY:
                            promote_kwargs_names.append(name)
                    pos += 1

            return ElementwiseTypePromotionRule(
                "aten",
                decorated_op.__name__,
                promote_args_positions=promote_args_positions,
                promote_kwargs_names=promote_kwargs_names,
                promotion_kind=type_promo_wrapper.type_promotion_kind,
            )

        logger.warning(
            "Cannot find type promotion rule for: %s.%s",
            decorated_op.__module__,
            decorated_op.__name__,
        )
        return None


class TypePromotionTable:
    """Type promotion table for torch.ops."""

    def __init__(self):
        self._rule_table = {}
        for rule in _GENERATED_ATEN_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)
        for rule in _EXTRA_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)

    @_beartype.beartype
    def add_rule(self, rule: TypePromotionRule) -> None:
        """Add a type promotion rule for a python op in a torch.ops module.

        Args:
            rule: Type promotion rule.
            module: Module containing the op. E.g. torch.ops.aten.

        Raises:
            ValueError: If the rule is invalid.
        """
        if not rule.is_valid():
            raise ValueError(f"Invalid type promotion rule: {rule}")
        self._rule_table[f"{rule.namespace}.{rule.op_name}"] = rule

    @_beartype.beartype
    def get_rule(
        self, py_op: torch._ops.OpOverloadPacket
    ) -> Optional[TypePromotionRule]:
        """Get type promotion rule for a python op under 'torch.ops.<namespace>'."""
        return self._rule_table.get(str(py_op), None)


@_beartype.beartype
def get_type_promotion_rule(
    diagnostic: diagnostics.Diagnostic,
    node: torch.fx.Node,
    type_promotion_table: TypePromotionTable,
) -> Optional[TypePromotionRule]:
    """Get type promotion rule for a node.

    Args:
        diagnostic: Diagnostic object.
        node: Node to get type promotion rule for.
        type_promotion_table: Type promotion table.

    Returns:
        Type promotion rule for the node. None if no rule is found or if the node is not
        representing a torch operator.
    """
    op = node.target
    if not isinstance(op, torch._ops.OpOverload):
        # TODO(bowbao): diagnostic.emit and diagnostic.set_message api.
        diagnostic.message = (
            f"Skipped for {diagnostics.format_argument(node)}: "
            f"node.target is not OpOverload. Got type: {type(op)}"
        )
        return None
    if (rule := type_promotion_table.get_rule(op.overloadpacket)) is None:
        diagnostic.message = (
            f"Skipped for {diagnostics.format_argument(node)}: "
            f"Cannot find type promotion rule for op: {op}"
        )
        return None

    diagnostic.info("Found type promotion rule: %s", rule)
    return rule


class _OpTraceDispatchMode(_python_dispatch.TorchDispatchMode):
    """Trace ops that were dispatched.

    Utilize the dispatch mechanism in [`__torch_dispatch__`](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557)
    to trace op overloads that were dispatched to. This is used to find the compatible
    op overload for a given op overload packet for different set of args and kwargs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traced_ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.traced_ops.append(func)
        return func(*args, **kwargs)


@_beartype.beartype
def find_compatible_op_overload(
    op: torch._ops.OpOverloadPacket, args: tuple, kwargs: dict
) -> torch._ops.OpOverload:
    """Find compatible OpOverload for an OpOverloadPacket using provided args and kwargs.

    Each "call_function" fx.Node in the fx.GraphModule has a target that represents a torch._ops.OpOverload.
    The OpOverload contains an OpOverloadPacket that holds all the available overloads for the operation.

    During the type promotion pass, there are cases where the types of the args and kwargs may change,
    such as promoting Python numbers to tensors. Consequently, the original OpOverload might not be
    compatible with the updated args and kwargs. This function is used to identify the compatible
    OpOverload for the given args and kwargs.

    Args:
        op: OpOverloadPacket to find compatible OpOverload for.
        args: The positional arguments to consider for compatibility.
        kwargs: The keyword arguments to consider for compatibility.

    Returns:
        torch._ops.OpOverload: The compatible OpOverload found for the given args and kwargs.

    Raises:
        RuntimeError: If no compatible op overload is found.

    Examples:
        >>> import torch
        >>> packet = torch.ops.aten.pow
        >>> args = (torch.tensor([1.0, 2.0]), 2)
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Scalar'
        >>> args = (torch.tensor([1.0, 2.0]), torch.tensor(2.0))
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Tensor'
    """
    # Utilize the dispatch mechanism to find the compatible op overload.
    op_trace_dispatch_mode = _OpTraceDispatchMode()
    with op_trace_dispatch_mode:
        op(*args, **kwargs)
    assert (
        len(op_trace_dispatch_mode.traced_ops) >= 1
    ), "Expected at least 1 traced op, got 0"

    new_op_overload = op_trace_dispatch_mode.traced_ops[0]
    assert isinstance(
        new_op_overload, torch._ops.OpOverload
    ), f"Expected OpOverload, got {type(new_op_overload)}"
    assert (
        new_op_overload.overloadpacket == op
    ), f"Expected same OpOverload packet, got {new_op_overload.overloadpacket} != {op}"

    return new_op_overload


class _TypePromotionInterpreter(torch.fx.Interpreter):
    """Interpreter that inserts type promotion for each node."""

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable,
    ):
        super().__init__(module)
        self.diagnostic_context = diagnostic_context
        self.type_promotion_table = type_promotion_table

    def _run_node_and_set_meta(self, node) -> Any:
        """Run node and set meta according to `fx_traceback.get_current_meta()`.

        This should be used on new nodes or nodes that have been modified.
        By default `Interpreter.run_node` does not update `node.meta`.
        Set `node.meta` to the current meta, except for `node.meta["val"]`, which is
        recomputed.
        """
        out = super().run_node(node)
        # Update interpreter env state with new output value.
        self.env[node] = out
        node.meta.update(
            (k, v)
            for k, v in fx_traceback.get_current_meta().items()
            if k not in node.meta
        )
        node.meta["val"] = proxy_tensor.extract_val(out)
        return out

    @_beartype.beartype
    def _create_node(
        self,
        graph: torch.fx.Graph,
        op_type: str,
        target: torch.fx.node.Target,
        args: tuple,
        kwargs: dict,
    ) -> torch.fx.Node:
        """Create a node and set its metadata."""
        assert op_type in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        ), f"Unexpected op_type: {op_type}"
        node = getattr(graph, op_type)(target, args, kwargs)
        self._run_node_and_set_meta(node)
        return node

    @_beartype.beartype
    def _rerun_node_after_type_promotion(
        self,
        diagnostic: diagnostics.Diagnostic,
        node: torch.fx.Node,
        expected_out_dtype: torch.dtype,
    ) -> None:
        """Rerun a node after type promotion and update node.meta["val"] with the output value."""
        node_val = node.meta.get("val", None)
        assert node_val is not None, f"Node {node} node.meta['val'] is not set."
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        target = node.target
        assert isinstance(
            target, torch._ops.OpOverload
        ), f"Expected OpOverload, got {type(target)}"
        node.target = find_compatible_op_overload(target.overloadpacket, args, kwargs)

        new_node_val = self._run_node_and_set_meta(node)
        assert isinstance(new_node_val, type(node_val)), (
            f"run_node output type should not change between runs. "
            f"Got {type(new_node_val)}, expect {type(node_val)}."
        )

        if isinstance(node_val, torch.Tensor):
            prev_node_dtype = node_val.dtype

            assert prev_node_dtype == expected_out_dtype, (
                f"node.meta['val'].dtype({prev_node_dtype}) does not agree with "
                f"type promotion rule({expected_out_dtype})."
            )

            if new_node_val.dtype != expected_out_dtype:
                # With explicit type promotion, the expected result dtype may not be
                # the same as the computation dtype. This is referred to as "op math".
                # We need to explicitly cast the output back to the expected dtype.
                # See more about "op math" topic at `_prims_common.elementwise_dtypes`.
                graph = node.graph
                with graph.inserting_after(node):
                    output_cast_node = self._create_node(
                        graph,
                        "call_function",
                        torch.ops.prims.convert_element_type.default,
                        (node,),
                        {"dtype": expected_out_dtype},
                    )
                    node.replace_all_uses_with(output_cast_node)
                    output_cast_node.args = (node,)
                    diagnostic.info(
                        "Node '%s' output dtype becomes %s due to op math. "
                        "Cast back to %s.",
                        node,
                        new_node_val.dtype,
                        expected_out_dtype,
                    )

        elif fx_type_utils.is_torch_symbolic_type(node_val):
            raise NotImplementedError(
                "Type promotion does not support node output of sym types."
            )
        elif isinstance(node_val, (list, tuple)):
            raise NotImplementedError(
                "Type promotion does not support node output of list or tuple."
            )
        else:
            raise RuntimeError(f"Unexpected node output type: {type(node_val)}.")

    @_beartype.beartype
    def _maybe_promote_arg(
        self,
        diagnostic: diagnostics.Diagnostic,
        node: torch.fx.Node,
        fx_arg: torch.fx.node.Argument,
        dtype: Optional[torch.dtype],
    ) -> torch.fx.node.Argument:
        """Promote fx_arg to dtype if necessary."""
        if dtype is None:
            diagnostic.info(
                "Argument %s is not promoted. Not mentioned by type promotion rule.",
                fx_arg,
            )
            return fx_arg

        if isinstance(fx_arg, torch.fx.Node):
            arg_val = self.env[fx_arg]
            if isinstance(arg_val, torch.Tensor):
                if (old_dtype := arg_val.dtype) != dtype:
                    # Promote tensor to dtype.
                    graph = node.graph
                    with graph.inserting_before(node):
                        diagnostic.info(
                            "Argument %s(%s) is promoted to %s.",
                            fx_arg,
                            old_dtype,
                            dtype,
                        )
                        return self._create_node(
                            graph,
                            "call_function",
                            torch.ops.prims.convert_element_type.default,
                            (fx_arg,),
                            {"dtype": dtype},
                        )
                diagnostic.info(
                    "Argument %s is not promoted. Already %s.", fx_arg, dtype
                )
                return fx_arg
            elif fx_type_utils.is_torch_symbolic_type(arg_val):
                arg_type = type(arg_val)
                equivalent_dtype = fx_type_utils.from_scalar_type_to_torch_dtype(
                    arg_type
                )
                assert equivalent_dtype is not None, f"Unexpected arg_type: {arg_type}"
                if equivalent_dtype != dtype:
                    # Promote Sym number to tensor of dtype.
                    graph = node.graph
                    with graph.inserting_before(node):
                        diagnostic.info(
                            "Argument %s(Scalar of equivalent dtype: %s) "
                            "is promoted to %s.",
                            fx_arg,
                            equivalent_dtype,
                            dtype,
                        )
                        return self._create_node(
                            graph,
                            "call_function",
                            torch.ops.aten.scalar_tensor.default,
                            (fx_arg,),
                            {"dtype": dtype},
                        )
                diagnostic.info(
                    "Argument %s is not promoted. Already %s.", fx_arg, dtype
                )
                return fx_arg
        elif (
            equivalent_dtype := fx_type_utils.from_scalar_type_to_torch_dtype(
                type(fx_arg)
            )
        ) is not None:
            if equivalent_dtype != dtype:
                # Promote number to tensor of dtype.
                # The op should have overload that supports tensor for this arg, otherwise
                # the type promotion rule should not suggest promoting this arg.
                graph = node.graph
                with graph.inserting_before(node):
                    diagnostic.info(
                        "Argument %s(Scalar of equivalent dtype: %s) "
                        "is promoted to %s.",
                        fx_arg,
                        equivalent_dtype,
                        dtype,
                    )
                    return self._create_node(
                        graph,
                        "call_function",
                        torch.ops.aten.scalar_tensor.default,
                        (fx_arg,),
                        {"dtype": dtype},
                    )
            diagnostic.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
            return fx_arg
        elif isinstance(fx_arg, (tuple, list)):
            diagnostic.info(
                "Argument %s is a tuple/list. Promoting each element.", fx_arg
            )
            return type(fx_arg)(
                self._maybe_promote_arg(diagnostic, node, fx_arg_elem, dtype)
                for fx_arg_elem in fx_arg
            )

        raise NotImplementedError(f"Unknown fx arg type: {type(fx_arg)}")

    @_beartype.beartype
    def _maybe_promote_node(
        self,
        diagnostic: diagnostics.Diagnostic,
        node: torch.fx.Node,
        rule: TypePromotionRule,
    ) -> torch.fx.Node:
        """Promote node inputs and outputs according to type promotion rule."""
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        type_promotion_info = rule.preview_type_promotion(args, kwargs)
        new_args = []
        new_kwargs = {}
        for i, arg in enumerate(node.args):
            new_args.append(
                self._maybe_promote_arg(
                    diagnostic, node, arg, type_promotion_info.args_dtypes.get(i, None)
                )
            )

        for name, arg in node.kwargs.items():
            new_kwargs[name] = self._maybe_promote_arg(
                diagnostic, node, arg, type_promotion_info.kwargs_dtypes.get(name, None)
            )
        new_args = tuple(new_args)

        if node.args != new_args or node.kwargs != new_kwargs:
            diagnostic.message = f"Applied type promotion for {node}. "
            node.args = new_args
            node.kwargs = new_kwargs
            self._rerun_node_after_type_promotion(
                diagnostic, node, type_promotion_info.out_dtype
            )
        else:
            diagnostic.message = f"Type promotion not needed for {node}. "

        return node

    @diagnostics.diagnose_call(
        rule=diagnostics.rules.fx_node_insert_type_promotion,
        level=diagnostics.levels.NONE,
    )
    def run_node(self, node: torch.fx.Node) -> Any:
        """This method is an override which inserts type promotion nodes as needed.

        For each `call_function` node, an initial check is conducted to determine if a type
        promotion rule is applicable. If a relevant rule exists, type casting nodes are
        introduced for the corresponding arguments. The OpOverload of the node is updated
        to one that accommodates the promoted types. Should the output type be different,
        type casting node is inserted for this output.

        The call `super().run_node(node)` is guaranteed to be invoked for each node.
        In the case of new or modified nodes, the result of `super().run_node(node)` is
        used to update its `node.meta["val"]` value.
        """
        diagnostic = self.diagnostic_context.inflight_diagnostic()
        with self._set_current_node(node):
            if node.op != "call_function":
                diagnostic.message = f"Skipped {node}: not a call_function."
            elif rule := get_type_promotion_rule(
                diagnostic, node, self.type_promotion_table
            ):
                self._maybe_promote_node(diagnostic, node, rule)

        return super().run_node(node)


class InsertTypePromotion(_pass.Transform):
    """Explicitly insert type promotion ops to the graph.

    This class subclasses `_pass.Transform` to provide graph level diagnostic tracking.
    Underneath, the main pass is driven by `_TypePromotionInterpreter`, which is a subclass
    of `torch.fx.Interpreter` to interpret the fx.Graph and perform the insertion of type
    promotion operations.

    The interpreter is extended with ability to track diagnostic information for each node.

    By re-running the new and modified nodes using the interpreter, we can update the
    metadata, specifically the fake tensor stored under node.meta["val"], and ensure it
    reflects the latest changes.

    See [FXE0015: fx_node_insert_type_promotion](https://pytorch.org/docs/master/generated/onnx_diagnostics_rules/FXE0015%3Afx-node-insert-type-promotion.html) for more details.  # noqa: B950
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        type_promotion_table: Optional[TypePromotionTable] = None,
    ):
        super().__init__(diagnostic_context, module)
        self.interpreter = _TypePromotionInterpreter(
            diagnostic_context, module, type_promotion_table or TypePromotionTable()
        )

    def _fetch_fake_args(self) -> Sequence[Optional[fake_tensor.FakeTensor]]:
        """Fetch fake args from fx graph.

        For each argument, try to fetch fake tensor from the matching placeholder node.
        """
        fake_args = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                try:
                    fake_tensor = _fake_tensor_from_node_val(node)
                except RuntimeError as e:
                    if not node.users:
                        # If the placeholder is not used, we can safely ignore it and put
                        # None as placeholder.
                        fake_tensor = None
                    else:
                        raise RuntimeError(
                            "Cannot fetch symbolic fake args from fx graph. "
                            "InsertTypePromotion pass needs to run with pre-existing fake args, "
                            "Otherwise the pass will produce inaccurate dynamic shape. "
                        ) from e

                fake_args.append(fake_tensor)
        return fake_args

    @_beartype.beartype
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        assert not args, (
            "`InsertTypePromotion` deduces symbolic fake arguments from the graph. "
            "It does not accept concrete arguments as input because this pass requires "
            "re-running the graph. When executed with newly faked concrete arguments, "
            "the pass loses the symbolic dynamic shape information."
        )
        assert not kwargs, "`kwargs` is not supported"

        fake_args = self._fetch_fake_args()
        fake_mode = self.fake_mode
        assert fake_mode is not None, "Cannot detect fake_mode."

        with proxy_tensor.maybe_disable_fake_tensor_mode(), (
            fake_mode
        ), fx_traceback.preserve_node_meta():
            self.interpreter.run(*fake_args)

        return self.module
