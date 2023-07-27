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
    Tuple,
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
        node: torch.fx.Node,  # this is used in diagnostic_message_formatter
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
            onnx_args: Arguments organized in PyTorch inputs way.
            onnx_kwargs: Keyword arguments organized in PyTorch inputs way.
            diagnostic_context: The diagnostic context to use for reporting errors.

            Returns:
                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
            Raises:
                RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        # TODO(justinchuby): Cache the OnnxSchemaChecker  so we don't need to run the init logic everytime
        overload_match_ranking: Dict[registration.SymbolicFunction, int] = {}
        diagnostic = diagnostic_context.inflight_diagnostic()

        # Iterate the overloaded functions in reverse order to prioritize the custom ones
        # over the default ones, and find the perfect match.
        for symbolic_function in reversed(default_and_custom_functions):
            function_opschema = _OnnxSchemaChecker(symbolic_function.onnx_function)

            if function_opschema.perfect_match_inputs(
                diagnostic, onnx_args, onnx_kwargs
            ):
                # If the perfect match is found, return the function
                return symbolic_function.onnx_function
            # Record the match score for the nearest match if it's not the perfect match
            overload_match_ranking[symbolic_function] = function_opschema.match_score

        # NOTE: If the perfect match is not found, find the nearest match
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
        # TODO: Should we have a better fallback mechanism?
        if function_group is None:
            function_group = self.onnx_registry.get_functions(
                namespace=internal_opname.namespace,
                op_name=internal_opname.op_name,
                overload=None,
            )
            if function_group is not None:
                # NOTE: Currently, most of torchlib functions are not registered with overload
                # in ONNX registry. So we will only log a warning in SARIF if we can't find the overload
                # to avoid spammy warnings in printout.
                # TODO: https://github.com/microsoft/onnxscript/issues/828
                op_full_name = internal_opname.qualified_name()
                diagnostic = diagnostic_context.inflight_diagnostic()
                diagnostic.with_additional_message(
                    "### The operator overload is not found in onnx registry!\n"
                    "Cannot find the operator overload in onnx registry, but "
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


class _OnnxSchemaChecker:
    """
    The OnnxSchemaChecker class is a checker for ONNX OpSchema and param schema.

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

    2. Optional dtype: dtype could be "unprovided". The difference from 2 is that dtype
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
        onnxfunction: The OnnxFunction.
        param_schema: The parameter schema defined in the OnnxFunction.
        op_schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.
        attributes: The attributes defined in the OpSchema.
        _matching_score: The matching score of the OnnxSchemaChecker .

    """

    def __init__(
        self,
        onnxfunction: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    ):
        """Initialize the OnnxSchemaChecker .

        Args:
            onnxfunction: The OnnxFunction.
        """
        self.onnxfunction = onnxfunction
        self.param_schema = self.onnxfunction.param_schemas()
        op_schema = self.onnxfunction.op_schema
        # Both `OnnxFunction` and `TracedOnnxFunction` never return None for `op_schema`.
        # However their base class would. Hence return type is annotated as Optional[OpSchema].
        assert op_schema is not None
        self.op_schema = op_schema
        self.type_constraints = {
            # "T": {"tensor(int64)"}
            constraint.type_param_str: set(constraint.allowed_type_strs)
            for constraint in self.op_schema.type_constraints
        }
        self.attributes = self.op_schema.attributes
        self._matching_score: int = 0

    @property
    def match_score(self) -> int:
        """The matching score of the OnnxSchemaChecker .

        Returns:
            The matching score of the OnnxSchemaChecker .
        """
        return self._matching_score

    @_beartype.beartype
    def perfect_match_inputs(
        self,
        diagnostic: diagnostics.Diagnostic,
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
            diagnostic: The diagnostic to use for logging detailed info.
            args: The input arguments organized in PyTorch inputs way.
            kwargs: The input keyword arguments organized in PyTorch inputs way.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """

        # NOTE: OnnxFunction does not have the same function signature as the original
        # PyTorch operator. We need to separate the input/attributes from the arguments.
        (
            function_inputs,
            function_attributes,
        ) = self._separate_input_attributes_from_arguments(
            self.param_schema,
            args,
            kwargs,
            fill_defaults=False,  # NOTE: We don't want to change inputs
        )

        # TODO(titaiwang): Currently the functions in torchlib are manully annotated,
        # so there are quite a few functions that wrongly annotated or strctly annotated.
        # The matching system relax the match while we fix them in the future.
        self._record_matching_score(function_inputs, function_attributes)

        diagnostic.with_additional_message("### Checking perfect match...\n")
        diagnostic.with_additional_message(
            f"{diagnostics.format_argument(self.onnxfunction)}"
        )
        diagnostic.with_additional_message(f"match score: {self.match_score}\n")

        if len(function_inputs) != len(self.op_schema.inputs):
            diagnostic.with_additional_message(
                f"#### Failed: input number mismatch! \n"
                f"Actual {len(function_inputs)} vs expected {len(self.op_schema.inputs)}\n"
            )
            return False
        for schema_input, torch_input in zip(self.op_schema.inputs, function_inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if not allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types isn't in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are not compatible
                diagnostic.with_additional_message(
                    f"#### Failed: input type mismatch for input '{schema_input.name}'! \n"
                    f"Actual {torch_input_compatible_types} vs\n"
                    f"expected {allowed_types}\n"
                )
                return False
        # Check attributes keys are the same
        if set(function_attributes) != set(self.attributes):
            # If the attributes of the OpSchema and the attributes don't match,
            # we know the function and the input are not compatible
            diagnostic.with_additional_message(
                f"#### Failed: attribute mismatch! \n"
                f"Actual {set(function_attributes)} vs\n"
                f"expected {set(self.attributes)}\n"
            )
            return False
        # Check attribute dtypes
        for attribute_name, attribute in function_attributes.items():
            if not self._match_onnx_attribute_type(attribute_name, attribute):
                # If the attribute type of the OpSchema and the attribute type don't match,
                # we know the function and the input are not compatible
                diagnostic.with_additional_message(
                    f"#### Failed: attribute '{attribute_name}' type mismatch! \n"
                    f"Actual {type(attribute)} vs\n"
                    f"expected {self.attributes[attribute_name].type}\n"
                )
                return False
        return True

    @_beartype.beartype
    def _match_onnx_attribute_type(
        self,
        attribute_name: str,
        attribute: fx_type_utils.Argument,
        is_sequence: bool = False,
    ) -> bool:
        if isinstance(attribute, (int, float, bool, str)):
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(
                type(attribute), is_sequence=is_sequence
            )
            if attribute_onnx_type != self.attributes[attribute_name].type:
                return False
        # If the attribute is an empty list, we don't know the type of the list
        # so it's a mismatch
        elif isinstance(attribute, (list, tuple)) and attribute:
            return self._match_onnx_attribute_type(
                attribute_name, attribute[0], is_sequence=True
            )
        else:
            # NOTE: Unrecognized attribute type
            return False
        return True

    @_beartype.beartype
    def _record_matching_score(
        self,
        inputs: Sequence[
            Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]
        ],
        attributes: Dict[str, fx_type_utils.Argument],
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
            inputs: The input arguments.
            attributes: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """

        # If they have different length of arguments, the score would be lower to those
        # functions which have the same length of arguments.
        for schema_input, torch_input in zip(self.op_schema.inputs, inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types is in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are compatible
                self._matching_score += 1
        # NOTE: The penalty is applied to those functions which have different attributes.
        for attribute_name, attribute_proto in self.attributes.items():
            if attribute_name not in attributes:
                # If the attribute of the OpSchema and the attribute don't match,
                # we know the function and the input are not compatible
                self._matching_score -= 1
                continue
            attribute = attributes[attribute_name]
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(
                type(attribute)
            )
            if attribute_onnx_type != attribute_proto.type:
                # If the attribute type of the OpSchema and the attribute type don't match,
                # we know the function and the input are not compatible
                self._matching_score -= 1
        # If there is any unexpected attribute in attributes, we know the function
        # and the input are not compatible
        extra_attrbute_counts = set(attributes).difference(set(self.attributes))
        self._matching_score -= len(extra_attrbute_counts)

    # NOTE: Referenced from onnxscript internal function.
    # Importing this function makes the code less robust, as it is not a public API.
    @_beartype.beartype
    def _separate_input_attributes_from_arguments(
        self,
        param_schemas: Sequence["onnxscript.values.ParamSchema"],
        args: Sequence[
            Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]
        ],
        kwargs: Dict[str, fx_type_utils.Argument],
        fill_defaults: bool = True,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Separate Python args and kwargs into ONNX inputs and attributes.

        Args:
            param_schemas: The parameter schemas of an Op or a OnnxFunction.
            args: The Python positional arguments supplied by the caller.
            kwargs: The Python keyword arguments supplied by the caller.
            fill_defaults: Whether to fill the default values for attributes.

        Returns:
            A tuple of two elements:
            - A list of ONNX inputs.
            - An dictionary of ONNX attribute names and values.

        Raises:
            TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
            TypeError: When a required input is not provided.
        """
        # args, kwargs and param_schemas should be all in order
        # user may not specify all inputs or attributes

        onnx_inputs: List[Any] = []
        onnx_attributes: Dict[str, Any] = dict()

        for i, param in enumerate(param_schemas):
            if param.is_variadic_input:
                # Exhaust all remaining args
                onnx_inputs.extend(args[i:])
                args = []
                continue
            if i < len(args):
                if param.is_input:
                    onnx_inputs.append(args[i])
                else:
                    onnx_attributes[param.name] = args[i]
            elif param.name in kwargs:
                if param.is_input:
                    onnx_inputs.append(kwargs[param.name])
                else:
                    onnx_attributes[param.name] = kwargs[param.name]
            elif param.is_attribute and param.default is not object():
                # User did not provide the attribute
                if fill_defaults:
                    onnx_attributes[param.name] = param.default

        return onnx_inputs, onnx_attributes


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
