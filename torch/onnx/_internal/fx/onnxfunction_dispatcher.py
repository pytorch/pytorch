"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import operator
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Set,
    TYPE_CHECKING,
    Union,
)

import torch
import torch._ops
import torch.fx
from torch.onnx import _constants
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    diagnostics,
    registration,
    type_utils as fx_type_utils,
)


if TYPE_CHECKING:
    import onnx.defs  # type: ignore[import]
    import onnxscript  # type: ignore[import]


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.


@runtime_checkable
class _TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen operators.

    It uses the `torch.ops` name to find the function. If not found, it falls back to default.
    Otherwise, the best match is found among all function overloads.
    An exact match has higher precedence over the closest ones.

    Below is a breakdown on how the dispatch mechanism work:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists.
        b. If not found, check ATen overload=default.

    2. Find the nearest match among all overloaded functions:
        a. If the types match perfectly, select the function.
        b. Otherwise, find the nearest one with the highest matching score. Because of
            the potential wrongly annotated dtypes and attributes matching, we use
            nearest match to find the best function once the aten name is targeted.

    NOTE: The nearest match `doesn't guarantee` a correct match, and a warning message is logged.
    """

    def __init__(
        self,
        onnx_registry: registration.OnnxRegistry,
        diagnostic_context: diagnostics.DiagnosticContext,
        opset_version: int = 18,
    ):
        """Initialize the ONNX Function dispatcher.

        Args:
            onnx_registry: The ONNX registry.
            diagnostic_context: The diagnostic context to use for reporting errors.
            opset_version: The ONNX opset version for the model.
        """
        self.onnx_registry = onnx_registry
        self.opset_version = opset_version
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        onnx_kwargs: Dict[str, fx_type_utils.Argument],
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]:
        """Dispatches an ONNX function based on the given FX node, arguments, and keyword arguments.
        Args:
            node: The TorchFX node to dispatch the function for.
            onnx_args: The arguments of the ONNX function.
            onnx_kwargs: The keyword arguments of the ONNX function.
            diagnostic_context: The diagnostic context to use for reporting errors.
        Returns:
            Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
        Raises:
            RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        aten_name = self.get_aten_name(node, diagnostic_context)
        # If there are no overloaded functions available for the given FX node, raise an
        # unsupported error
        function_overloads = self.get_function_overloads(
            node, aten_name, diagnostic_context
        )
        # If there are overloaded functions available, we will find one that perfect or
        # nearest matches the given arguments and keyword arguments
        return self._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            aten_name,
            function_overloads,
            onnx_args,
            onnx_kwargs,
            diagnostic_context,
        )

    @_beartype.beartype
    def _find_the_perfect_or_nearest_match_onnxfunction(
        self,
        node: torch.fx.Node,
        aten_name: str,
        function_overloads: Set[
            Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]
        ],
        onnx_args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        onnx_kwargs: Dict[str, fx_type_utils.Argument],
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments."""
        overload_match_ranking: Dict[
            Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction], int
        ] = {}
        # TODO(justinchuby): Cache the OpSchemaWrapper so we don't need to run the init logic everytime
        for overload_func in function_overloads:
            function_opschema = _OpSchemaWrapper(overload_func.op_schema)
            if function_opschema.perfect_match_inputs(onnx_args, onnx_kwargs):
                # If the perfect match is found, return the function
                return overload_func
            # Record the match score for the nearest match if it's not the perfect match
            overload_match_ranking[overload_func] = function_opschema.match_score

        # TODO(titaiwang): Change inflight_diagnostic to a new rule. Do we need to handle
        # the special case where the same scores happended?
        # If the perfect match is not found, find the nearest match
        warnings.warn(
            f"A perfect matched Opchema is not found in torchlib for {aten_name}, but \n"
            f"a nearest match is found. Please check the ONNX output carefully. \n",
        )
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.WARNING,
            f"Cannot find a perfect match of symbolic overload for {aten_name}, "
            f"which should be registered under {node.target}. But a nearest match is found.",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        return max(overload_match_ranking, key=overload_match_ranking.get)  # type: ignore[arg-type]

    @_beartype.beartype
    def get_aten_name(
        self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext
    ) -> str:
        """Get the aten name from the target.

        Args:
            node: The TorchFX node to get the aten name for.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            The aten name of the given node.
        """
        if node.target == operator.getitem:
            return "aten::getitem"
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            # aten::sym_size is the only OverloadPacket that we support.
            # schema: aten::sym_size(Tensor self, int dim) -> Tensor
            if node.target != torch.ops.aten.sym_size:
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                    diagnostics.rules.no_symbolic_function_for_call_function,
                    diagnostics.levels.ERROR,
                    f"Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!",
                    unsupported_fx_node=node,
                )
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
            # overloadpacket for some reasons.
            # https://github.com/pytorch/pytorch/issues/97201
            aten_op_default = node.target.default
            return aten_op_default.name()  # type: ignore[attr-defined]

        if _symint_symfloat_builtin_to_exporter_key_table(node.target) is not None:
            # Make sure it's symint/symfloat consuming builtin ops.
            for node_arg in node.args:
                if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                    isinstance(node_arg, torch.fx.Node)
                    and not isinstance(
                        node_arg.meta["val"], (torch.SymInt, torch.SymFloat)
                    )
                ):
                    # TODO: reduce number of explicit initializations.
                    # TODO: Log location, stack.
                    diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                        diagnostics.rules.no_symbolic_function_for_call_function,
                        diagnostics.levels.ERROR,
                        f"Unsupported node arg: {node_arg} (type {type(node_arg)}) with builtin function: {node.target},"
                        " only int/float/SymInt/SymFloat is supported with built-in ops!",
                        unsupported_fx_node=node,
                    )
                    diagnostic_context.log(diagnostic)
                    raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            aten_op = _symint_symfloat_builtin_to_exporter_key_table(node.target)
            return aten_op.name()  # type: ignore[attr-defined]
        if isinstance(node.target, torch._ops.OpOverload):
            return node.target.name()

        # Unexpected target, raise error.
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Unknown call_function target: {node.target}",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

    @_beartype.beartype
    def get_function_overloads(
        self,
        node: torch.fx.Node,
        aten_name: str,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Set[Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]]:
        """Get the function overloads from the registry.

        Args:
            node: The node to get the function overloads for.
            aten_name: The aten name of the node.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            Set of function overloads.
        """
        function_group = None

        if self.onnx_registry.is_registered_op(aten_name, self.opset_version):
            function_group = self.onnx_registry.get_function_group(aten_name)

        # Fall back to overloadpacket name: eg: aten.add.Tensor -> aten::add
        elif hasattr(
            node.target, "overloadpacket"
        ) and self.onnx_registry.is_registered_op(
            node.target.overloadpacket._qualified_op_name,  # type: ignore[union-attr]
            self.opset_version,
        ):
            function_group = self.onnx_registry.get_function_group(
                node.target.overloadpacket._qualified_op_name  # type: ignore[union-attr]
            )

        if function_group is not None:
            # TODO(titaiwang): dispatch opset version.
            dispatched_version = _dispatch_opset_version(
                self.opset_version, function_group.support_opset()
            )
            if dispatched_version is not None:
                function_overloads = function_group.get(dispatched_version)
                if function_overloads is not None:
                    return function_overloads

        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Cannot find symbolic function for {aten_name}, "
            f"which should be registered under {node.target}.",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)


@_beartype.beartype
def _symint_symfloat_builtin_to_exporter_key_table(
    target,
) -> Optional[torch._ops.OpOverload]:
    """Maps builtin ops to exporter key table."""

    _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE: Dict[
        Union[Callable[..., Any], str], torch._ops.OpOverload
    ] = {
        operator.mul: torch.ops.aten.mul.default,  # type: ignore[has-type]
        operator.add: torch.ops.aten.add.default,  # type: ignore[has-type]
        operator.pow: torch.ops.aten.pow.int,  # type: ignore[has-type]
        operator.sub: torch.ops.aten.sub.default,  # type: ignore[has-type]
    }
    return _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE.get(target)


@_beartype.beartype
def _dispatch_opset_version(
    target: int, registered_opsets: Collection[int]
) -> Optional[int]:
    """Finds the registered opset given a target opset version and the available opsets.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    if not registered_opsets:
        return None

    descending_registered_versions = sorted(registered_opsets, reverse=True)
    # Linear search for the opset version, which is fine since the number of opset
    # versions is small.

    if target >= _constants.ONNX_BASE_OPSET:
        # Always look down toward opset 1 when the target is >= ONNX_BASE_OPSET (opset 9).
        # When a custom op is register at opset 1, we want to be able to discover it as a
        # fallback for all opsets >= ONNX_BASE_OPSET.
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None

    # target < opset 9. This is the legacy behavior to support opset 7 and opset 8.
    # for caffe2 support. We search up toward opset 9.
    for version in reversed(descending_registered_versions):
        # Count back up until _constants.ONNX_BASE_OPSET
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version

    return None


class _OpSchemaWrapper:
    """
    The OpSchemaWrapper class is a wrapper for ONNX OpSchema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types.

    There are three types of ONNX overloads in torchlib:

    1. Different types: Caused by the difference between the ONNX spec and PyTorch.The
        matching system finds the correct one.

        ```python
        @torch_op("aten::mul")
        def aten_mul(self: TReal, other: TReal) -> TReal:
            ...

        @torch_op("aten::mul")
        def aten_mul_bool(self: BOOL, other: BOOL) -> BOOL:
            ...
    ```

    2. Optional dim: caused by unsupported op.OptionalHasElement (will support on opset
        version == 20). dim could be "None"

        ```python
        @torch_op("aten::argmax", trace_only=True)
        def aten_argmax(
            self: TrealOrUInt8, dim: Optional[int] = None, keepdim: bool = False
        ) -> TrealOrUInt8:
            ...

        @torch_op("aten::argmax", private=True)
        def _aten_argmax_dim(self: TrealOrUInt8, dim: int, keepdim: bool = False) -> TrealOrUInt8:
            ...
        ```

        This case is impossible to differentiate, as they both might have dim in kwargs, so
        in this case, please make sure you turn the one with `dim: int` to private function.

    3. Optional dtype: dtype could be "unprovided". The difference from 2 is that dtype
        would not be None.

        ```python
        @torch_op("aten::new_full")
        def aten_new_full(self: TTensor, size: INT64, fill_value: TTensor) -> TTensor:
            ...

        @torch_op("aten::new_full")
        def aten_new_full_dtype(self: TTensor, size: INT64, fill_value: TTensor, dtype: int) -> TTensor:
            ...
        ```

        Depends on dtype is provided or not, matching system will dispatch the ATen op to
        the correct one.

    Attributes:
        schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.

    """

    def __init__(self, op_schema: onnx.defs.OpSchema):
        """Initialize the OpSchemaWrapper.

        Args:
            op_schema: The ONNX OpSchema.
        """
        self.schema = op_schema
        self.type_constraints = {
            # "T": {"tensor(int64)"}
            constraint.type_param_str: set(constraint.allowed_type_strs)
            for constraint in op_schema.type_constraints
        }
        # FIXME(titaiwang): Need AttributeProto to support get default_value.
        # TODO(titaiwang): attribut type is not checked.
        self.attributes = set(op_schema.attributes)
        self._matching_score: int = 0

    @property
    def match_score(self) -> int:
        """The matching score of the OpSchemaWrapper.

        Returns:
            The matching score of the OpSchemaWrapper.
        """
        return self._matching_score

    @_beartype.beartype
    def perfect_match_inputs(
        self,
        args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        kwargs: Dict[str, fx_type_utils.Argument],
    ) -> bool:
        """Check if the inputs perfectly match the OpSchema requirements.

        The definition of perfect match is that the input types are all in the type
        constraints and the number of inputs matches the number of inputs in the
        OpSchema.

        Args:
            args: The input arguments.
            kwargs: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        # TODO(titaiwang): Currently the functions in torchlib are manully annotated,
        # so there are quite a few functions that wrongly annotated or strctly annotated.
        # The matching system relax the match while we fix them in the future.
        self._record_matching_score(args, kwargs)

        # TODO: Refine the logic for it to be more robust
        # TODO: Handle attributes
        if len(args) != len(self.schema.inputs):
            return False
        for schema_input, torch_input in zip(self.schema.inputs, args):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if not allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types isn't in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are not compatible
                return False
        if set(kwargs) != self.attributes:
            # If the attributes of the OpSchema and the kwargs don't match,
            # we know the function and the input are not compatible
            return False
        return True

    @_beartype.beartype
    def _record_matching_score(
        self,
        args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        kwargs: Dict[str, fx_type_utils.Argument],
    ):
        """Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        How the matchsing score is calculated:
        1. score += 1 if one input type is in the type constraints.
        2. score -= 1 if one kwarg is not symmetrically the same.

        Limitations:
        1. An Overload is punished if it doesn't have `default` attributes.
        2. None/NoeType/[] could result in zero matches, and the same score of overloads,
            which will be recorded in SARIF.

        Args:
            args: The input arguments.
            kwargs: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """

        # If they have different length of arguments, the score would be lower to those
        # functions which have the same length of arguments.
        for schema_input, torch_input in zip(self.schema.inputs, args):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types is in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are compatible
                self._matching_score += 1
        # The penalty is applied to those functions which have different attributes.
        diff = self.attributes.symmetric_difference(set(kwargs))
        self._matching_score -= len(diff)


@_beartype.beartype
def _find_onnx_data_type(
    torch_input: Optional[Union[_TensorLike, str, int, float, bool, list, tuple]]
) -> Set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    if isinstance(torch_input, _TensorLike) and torch_input.dtype is not None:
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(torch_input.dtype)
    if isinstance(torch_input, (int, float, bool, str)):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(type(torch_input))
    if isinstance(torch_input, (list, tuple)) and torch_input:  # [Tensor, Tensor]
        set_dtype = _find_onnx_data_type(torch_input[0])
        if any(isinstance(input, _TensorLike) for input in torch_input):
            # NOTE: Any Tensor involved in a list would make it a seq(tensor(onnx_type))
            return {f"seq({dtype})" for dtype in set_dtype}
        else:
            # constant list of non-tensor type
            return set_dtype
    if (
        torch_input is None
        or (isinstance(torch_input, _TensorLike) and torch_input.dtype is None)
        or (isinstance(torch_input, (list, tuple)) and not torch_input)
    ):
        # NOTE: None, No dtype, and empty list are edge cases, we allow it to be any type to relax the type check
        # seq(tensor) also goes to here, as it is not supported in torchscript, and it would be None in this case.
        return set()

    raise RuntimeError(f"Unknown input type from input: {torch_input}")
