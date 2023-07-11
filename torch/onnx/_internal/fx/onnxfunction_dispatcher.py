"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import operator
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
    Union,
)

import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    diagnostics,
    registration,
    type_utils as fx_type_utils,
)


if TYPE_CHECKING:
    import onnx.defs  # type: ignore[import]
    import onnxscript  # type: ignore[import]


@_beartype.beartype
def _find_opschema_matched_symbolic_function_disagnostic_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    default_and_custom_functions: List[registration.SymbolicFunction],
    *args,
    **kwargs,
) -> str:
    """Format the diagnostic message for the nearest match warning."""
    all_function_overload_names = ""
    for symbolic_func in default_and_custom_functions:
        overload_func = symbolic_func.onnx_function
        all_function_overload_names += f"ONNX Node: {overload_func.name}[opset={overload_func.opset};is_custom={symbolic_func.is_custom}]. \n"  # noqa: B950
    return f"FX Node: {node.target}. \n" f"{all_function_overload_names}"


@_beartype.beartype
def _find_operator_overloads_in_onnx_registry_disagnostic_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    """Format the diagnostic message for the nearest match warning."""
    return f"Searching operator overload: '{node.target}' in onnx registry...\n"


class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen operators.

    It uses the `torch.ops` name to find the function. If not found, it falls back to default.
    Otherwise, the best match is found among all function overloads. An exact match has
    higher precedence over the closest ones.

    Below is a breakdown on how the dispatch mechanism works:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists in the registry.
        b. If not, check if the default overload exists in the registry.

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
    ):
        """Initialize the ONNX Function dispatcher.

        Args:
            onnx_registry: The ONNX registry.
            diagnostic_context: The diagnostic context to use for reporting errors.
        """
        self.onnx_registry = onnx_registry
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence[
            Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]
        ],
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
        # If there are no overloaded functions available for the given FX node, raise an
        # unsupported error
        default_and_custom_functions = self.get_function_overloads(
            node, diagnostic_context
        )

        # If there are overloaded functions available, we will find one that perfect or
        # nearest matches the given arguments and keyword arguments
        return self._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            default_and_custom_functions,
            onnx_args,
            onnx_kwargs,
            diagnostic_context,
        )

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.find_opschema_matched_symbolic_function,
        diagnostic_message_formatter=_find_opschema_matched_symbolic_function_disagnostic_message_formatter,
    )
    def _find_the_perfect_or_nearest_match_onnxfunction(
        self,
        node: torch.fx.Node,
        default_and_custom_functions: List[registration.SymbolicFunction],
        onnx_args: Sequence[
            Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]
        ],
        onnx_kwargs: Dict[str, fx_type_utils.Argument],
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments.

        Args:
            default_and_custom_functions: The list includes overloaded functions, with
                custom ones appearing after the default ones.
            onnx_args: The arguments of the ONNX function.
            onnx_kwargs: The keyword arguments of the ONNX function.
            diagnostic_context: The diagnostic context to use for reporting errors.

            Returns:
                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
            Raises:
                RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        # TODO(justinchuby): Cache the OpSchemaWrapper so we don't need to run the init logic everytime
        overload_match_ranking: Dict[registration.SymbolicFunction, int] = {}

        # Iterate the overloaded functions in reverse order to prioritize the custom ones
        # over the default ones, and find the perfect match.
        for symbolic_function in reversed(default_and_custom_functions):
            overload_func = symbolic_function.onnx_function
            function_opschema = _OpSchemaWrapper(overload_func.op_schema)
            if function_opschema.perfect_match_inputs(onnx_args, onnx_kwargs):
                # If the perfect match is found, return the function
                return overload_func
            # Record the match score for the nearest match if it's not the perfect match
            overload_match_ranking[symbolic_function] = function_opschema.match_score

        # NOTE: If the perfect match is not found, find the nearest match
        diagnostic = diagnostic_context.inflight_diagnostic()
        diagnostic.with_additional_message(
            "### Exact match is not found!\n"
            "Cannot find a perfect match of symbolic overload, "
            "a nearest match is found. Please check the ONNX output carefully. \n",
        )
        diagnostic.level = diagnostics.levels.WARNING

        # NOTE: Tie breaker: if there are multiple nearest matches, we will choose the one
        # that is custom first
        symbolic_function_list: List[registration.SymbolicFunction] = sorted(
            overload_match_ranking,
            key=lambda k: (overload_match_ranking[k], k.is_custom),
            reverse=True,
        )
        return symbolic_function_list[0].onnx_function

    @_beartype.beartype
    def _get_aten_name(
        self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext
    ) -> registration.OpName:
        """Get the OpName from the target.

        Args:
            node: The TorchFX node to get the aten name for.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            The internal op name within dataclass: registration.OpName.
        """
        if node.target == operator.getitem:
            return registration.OpName.from_name_parts(
                namespace="aten", op_name="getitem"
            )
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
            return registration.OpName.from_op_overload(op_overload=aten_op_default)  # type: ignore[no-any-return]

        if (
            aten_op := _symint_symfloat_builtin_to_exporter_key_table(node.target)
        ) is not None:
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
            return registration.OpName.from_op_overload(op_overload=aten_op)

        if isinstance(node.target, torch._ops.OpOverload):
            return registration.OpName.from_op_overload(op_overload=node.target)

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
    @diagnostics.diagnose_call(
        diagnostics.rules.find_operator_overloads_in_onnx_registry,
        diagnostic_message_formatter=_find_operator_overloads_in_onnx_registry_disagnostic_message_formatter,
    )
    def get_function_overloads(
        self,
        node: torch.fx.Node,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> List[registration.SymbolicFunction]:
        """Get the function overloads from the registry.

        Args:
            node: The node to get the function overloads for.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            The list contains SymbolicFunctions, starting with the default ones and
            followed by any custom ones.
        """

        internal_opname: registration.OpName = self._get_aten_name(
            node=node, diagnostic_context=diagnostic_context
        )

        # NOTE: If the ATen/Custom operators are not registered, the group will be None.
        # And non-registerd ATen/Custom operators will trigger error in the next step.
        function_group: Optional[List[registration.SymbolicFunction]] = None

        function_group = self.onnx_registry.get_functions(
            namespace=internal_opname.namespace,
            op_name=internal_opname.op_name,
            overload=internal_opname.overload,
        )

        # NOTE: Fall back to default overload if the ONNX registry doesn't have the overload.
        if function_group is None:
            function_group = self.onnx_registry.get_functions(
                namespace=internal_opname.namespace,
                op_name=internal_opname.op_name,
                overload=None,
            )

            # NOTE: Currently, most of torchlib functions are not registered with overload
            # in ONNX registry. So we will only log a warning in SARIF if we can't find the overload
            # to avoid spammy warnings in printout.
            # TODO: https://github.com/microsoft/onnxscript/issues/828
            op_full_name = internal_opname.qualified_name()
            diagnostic = diagnostic_context.inflight_diagnostic()
            diagnostic.with_additional_message(
                "### The operator overload is not found in onnx registry!\n"
                "Cannot find the operator overload in onnx registry, but"
                "the default overload is found. Please check the ONNX output carefully. \n",
            )
            diagnostic.level = diagnostics.levels.WARNING

        # NOTE: If the ATen/Custom operators are not registered, the group will be None.
        if function_group is not None:
            return function_group

        # If we can't find the function group, raise error.
        op_full_name = internal_opname.qualified_name()
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Cannot find symbolic function for {op_full_name}, "
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
        args: Sequence[
            Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]
        ],
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
        args: Sequence[
            Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]
        ],
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
    torch_input: Optional[
        Union[fx_type_utils.TensorLike, str, int, float, bool, list, tuple]
    ]
) -> Set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    if (
        isinstance(torch_input, fx_type_utils.TensorLike)
        and torch_input.dtype is not None
    ):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(torch_input.dtype)
    if isinstance(torch_input, (int, float, bool, str)):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(type(torch_input))
    if isinstance(torch_input, (list, tuple)) and torch_input:  # [Tensor, Tensor]
        set_dtype = _find_onnx_data_type(torch_input[0])
        if any(isinstance(input, fx_type_utils.TensorLike) for input in torch_input):
            # NOTE: Any Tensor involved in a list would make it a seq(tensor(onnx_type))
            return {f"seq({dtype})" for dtype in set_dtype}
        else:
            # constant list of non-tensor type
            return set_dtype
    if (
        torch_input is None
        or (
            isinstance(torch_input, fx_type_utils.TensorLike)
            and torch_input.dtype is None
        )
        or (isinstance(torch_input, (list, tuple)) and not torch_input)
    ):
        # NOTE: None, No dtype, and empty list are edge cases, we allow it to be any type to relax the type check
        # seq(tensor) also goes to here, as it is not supported in torchscript, and it would be None in this case.
        return set()

    raise RuntimeError(f"Unknown input type from input: {torch_input}")
