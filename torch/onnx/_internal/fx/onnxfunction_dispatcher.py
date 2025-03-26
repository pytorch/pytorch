# mypy: allow-untyped-defs
"""Dispatcher for AtenLib functions from onnx-script.

This is a deprecated module to be removed.
"""

from __future__ import annotations

import logging
import operator
import types
from typing import Any, TYPE_CHECKING

import torch
import torch._ops
import torch.fx
from torch.onnx._internal.fx import registration, type_utils as fx_type_utils


if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnxscript  # type: ignore[import]
    from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
        graph_building as onnxscript_graph_building,
    )

    from torch.onnx._internal._exporter_legacy import OnnxRegistry


logger = logging.getLogger(__name__)


class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen/Custom operators.

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

    3. Tie-breaker: If there are multiple nearest matches, we will select the one with
        the highest matching score.

    NOTE: The nearest match `doesn't guarantee` a correct match, and a warning message is logged.
    """

    def __init__(
        self,
        onnx_registry: OnnxRegistry,
    ):
        """Initialize the ONNX Function dispatcher.

        Args:
            onnx_registry: The ONNX registry.
        """
        self.onnx_registry = onnx_registry

    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence[
            fx_type_utils.TensorLike | str | int | float | bool | list | complex | None
        ],
        onnx_kwargs: dict[str, fx_type_utils.Argument],
    ) -> onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction:
        """Dispatches an ONNX function based on the given FX node, arguments, and keyword arguments.
        Args:
            node: The TorchFX node to dispatch the function for.
            onnx_args: The arguments of the ONNX function.
            onnx_kwargs: The keyword arguments of the ONNX function.

        Returns:
            Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
        Raises:
            RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        # If there are no overloaded functions available for the given FX node, raise an
        # unsupported error
        default_and_custom_functions = self.get_function_overloads(node)

        # If there are overloaded functions available, we will find one that perfect or
        # nearest matches the given arguments and keyword arguments
        return self._find_the_perfect_or_nearest_match_onnxfunction(
            node,
            default_and_custom_functions,
            onnx_args,
            onnx_kwargs,
        )

    def _filter_or_keep_complex(
        self,
        node,
        default_and_custom_functions: list[registration.ONNXFunction],
    ) -> list[registration.ONNXFunction]:
        """Filter the complex functions if the input has complex dtype."""

        args_with_complex_dtype = [_is_arg_with_complex_dtype(arg) for arg in node.args]
        if any(args_with_complex_dtype):
            default_and_custom_functions = [
                func for func in default_and_custom_functions if func.is_complex
            ]
            # If we can't find the complex function group, raise error.
            if not default_and_custom_functions:
                op_full_name = self._get_aten_name(node).qualified_name()
                raise RuntimeError(
                    f"Cannot find any COMPLEX symbolic function for {op_full_name}, "
                    f"which should be registered under {node.target}.",
                )
        else:
            default_and_custom_functions = [
                func for func in default_and_custom_functions if not func.is_complex
            ]
            # If we can't find the complex function group, raise error.
            if not default_and_custom_functions:
                op_full_name = self._get_aten_name(node).qualified_name()
                raise RuntimeError(
                    f"Can ONLY find COMPLEX symbolic function for {op_full_name}, "
                    f"which should be registered under {node.target}.",
                )
        return default_and_custom_functions

    def _find_the_perfect_or_nearest_match_onnxfunction(
        self,
        node: torch.fx.Node,
        default_and_custom_functions: list[registration.ONNXFunction],
        onnx_args: Sequence[
            fx_type_utils.TensorLike | str | int | float | bool | list | complex | None
        ],
        onnx_kwargs: dict[str, fx_type_utils.Argument],
    ):
        """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments.

        Args:
            default_and_custom_functions: The list includes overloaded functions, with
                custom ones appearing after the default ones.
            onnx_args: Arguments organized in PyTorch inputs way.
            onnx_kwargs: Keyword arguments organized in PyTorch inputs way.

            Returns:
                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
            Raises:
                RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        overload_match_ranking: dict[registration.ONNXFunction, int | None] = {}

        # Iterate the overloaded functions in reverse order to prioritize the custom ones
        # over the default ones, and find the perfect match.
        for symbolic_function in reversed(default_and_custom_functions):
            function_opschema = _OnnxSchemaChecker(symbolic_function.onnx_function)

            # NOTE: 1. If the perfect match is found, return the function
            if function_opschema.perfect_match_inputs(onnx_args, onnx_kwargs):
                return symbolic_function.onnx_function
            # Record the match score for the nearest match if it's not the perfect match
            overload_match_ranking[symbolic_function] = function_opschema.match_score

        # NOTE: 2. If there is no perfect match, find the nearest match among the nearest matche candidates
        # If there is no nearest match, raise an error
        overload_match_ranking = {
            k: v for k, v in overload_match_ranking.items() if v is not None
        }
        if not overload_match_ranking:
            # If there are no overloaded functions available for the given FX node, raise an
            # unsupported error
            op_full_name = self._get_aten_name(node).qualified_name()
            raise RuntimeError(
                f"Cannot find any perfect/nearest match of symbolic function for {op_full_name},"
                f"which should be registered under {node.target}.",
            )

        # NOTE: 3. Tie breaker: if there are multiple nearest matches, we will choose the one
        # that is custom first. If there are multiple custom ones, we will choose the one
        # that is added lastly in the list.
        symbolic_function_list: list[registration.ONNXFunction] = sorted(
            overload_match_ranking,
            key=lambda k: (
                overload_match_ranking[k],
                k.is_custom,
                default_and_custom_functions.index(k),
            ),
            reverse=True,
        )
        return symbolic_function_list[0].onnx_function

    def _get_aten_name(self, node: torch.fx.Node) -> registration.OpName:
        """Get the OpName from the target.

        Args:
            node: The TorchFX node to get the aten name for.

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
                raise RuntimeError(
                    f"Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!",
                )
            # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
            # overloadpacket for some reasons.
            # https://github.com/pytorch/pytorch/issues/97201
            aten_op_default = node.target.default
            return registration.OpName.from_op_overload(op_overload=aten_op_default)  # type: ignore[no-any-return]

        if isinstance(node.target, types.BuiltinFunctionType):
            # Make sure it's symint/symfloat consuming builtin ops.
            for node_arg in node.args:
                if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                    isinstance(node_arg, torch.fx.Node)
                    and not fx_type_utils.is_torch_symbolic_type(node_arg.meta["val"])
                ):
                    raise RuntimeError(
                        f"Unsupported node arg: {node_arg} (type {type(node_arg)}) with builtin function: {node.target},"
                        " only int/float/SymInt/SymFloat is supported with built-in ops!",
                    )
            return registration.OpName.from_builtin_function(node.target)

        if isinstance(node.target, torch._ops.OpOverload):
            return registration.OpName.from_op_overload(op_overload=node.target)

        # Unexpected target, raise error.
        raise RuntimeError(f"Unknown call_function target: {node.target}")

    def get_function_overloads(
        self,
        node: torch.fx.Node,
    ) -> list[registration.ONNXFunction]:
        """Get the function overloads from the registry.

        Args:
            node: The node to get the function overloads for.

        Returns:
            The list contains ONNXFunctions, starting with the default ones and
            followed by any custom ones.
        """

        internal_opname: registration.OpName = self._get_aten_name(node=node)

        # If the ATen/Custom operators are not registered, the group will be None.
        # And non-registered ATen/Custom operators will trigger error in the next step.
        function_group: list[registration.ONNXFunction] | None = None

        function_group = self.onnx_registry.get_op_functions(
            namespace=internal_opname.namespace,
            op_name=internal_opname.op_name,
            overload=internal_opname.overload,
        )

        # NOTE: Fall back to default overload if the ONNX registry doesn't have the overload.
        if function_group is None:
            function_group = self.onnx_registry.get_op_functions(
                namespace=internal_opname.namespace,
                op_name=internal_opname.op_name,
                overload=None,
            )
            if function_group is not None:
                op_full_name = internal_opname.qualified_name()

        if function_group is not None:
            # NOTE: If the input has complex dtype, we will only dispatch to the complex functions.
            function_group = self._filter_or_keep_complex(node, function_group)
            return function_group  # type: ignore[return-value]

        op_full_name = internal_opname.qualified_name()
        raise RuntimeError(
            f"Cannot find symbolic function for {op_full_name}, "
            f"which should be registered under {node.target}.",
        )


class _OnnxSchemaChecker:
    """
    The OnnxSchemaChecker class is a checker for ONNX OpSchema and param schema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types. A function will be evaluated as perfect match, nearest match eligible,
    or no match.

    Here are some common examples in categories:

    1. [NOTE: Perfect match]: The number of inputs and attributes are exactly the same as
        the OpSchema. The types of inputs and attributes are exactly the same as the
        OpSchema.

        ```python
        inputs = (Tensor[2, 3], Tensor[2, 3])
        attributes = {"alpha": 1.0}


        @torch_op("aten::op")
        def aten_op(self: TReal, other: TReal, alpha: float = 1) -> TReal: ...
        ```
        Result: Perfect match.

    2. [NOTE: Optional input]: The dispatcher recognizes optional inputs. However,
        the input can't be ignored. None must be provided.

        ```python
        inputs = (Tensor([2, 3]), None)
        attributes = {}

        aten_op(X: TTensor, Y: Optional[INT64]):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::convolution`.

    3. [NOTE: Different attributes]: If an attribute is provided with value, it's
        a must to match the attribute in function signature.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a":1, "b":2}

        aten_op(X: TTensor, a: int):
            ...
        ```
        Result: No match.
        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    4. [NOTE: Default attributes]: Default attribute will fill in the value into
        inputs/attributes.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::clone`

    5. [NOTE: Ignore attribute with None value]: The attributes with None value
        will be ignored in matching.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor):
            ...
        ```
        Result: Perfect match.

        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Nearest match eligible.

        Real example: `aten::div` vs `aten::div.Tensor_mode`.

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
        onnxfunction: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction,
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
        self._matching_score: int | None = None

    @property
    def match_score(self) -> int | None:
        """The matching score of the OnnxSchemaChecker .

        If this remains None, it means the matching score has not been calculated,
        and it's not a nearest match candidate.

        Returns:
            The matching score of the OnnxSchemaChecker .
        """
        return self._matching_score

    def perfect_match_inputs(
        self,
        args: Sequence[
            fx_type_utils.TensorLike | str | int | float | bool | list | complex | None
        ],
        kwargs: dict[str, fx_type_utils.Argument],
    ) -> bool:
        """Check if the inputs perfectly match the OpSchema requirements.

        The definition of perfect match is that the input types are all in the type
        constraints and the number of inputs matches the number of inputs in the
        OpSchema.

        Checking steps:
        1. The function signature matches the inputs number, and attribute names.
        2. The input/attribute types are all in the type constraints.

        A function should at least pass the first step to be eligible for the
        nearest matching.

        Args:
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
            fill_defaults=True,  # fill defaults for optional arguments to match
        )
        # NOTE: 1. Check if the input number and attribute names match the
        # OpSchema. If it's not, we know the function is not eligible to be a perfect
        # match, nor a nearest match.
        # We use is_perfect_match to postpone the return value to the end
        # of the function, as we want to log all the mismatch info.
        is_perfect_match = True
        if len(function_inputs) != len(self.op_schema.inputs):
            logger.info(
                "Actual %d vs expected %d",
                len(function_inputs),
                len(self.op_schema.inputs),
            )
            logger.info("The function is not a nearest match candidate.")
            is_perfect_match = False

        if set(function_attributes) != set(self.attributes):
            logger.info("The function is not a nearest match candidate.")
            is_perfect_match = False

        # If it's already not a perfect match, we can return False directly. Further
        # checking is only for the functions that are eligible for nearest match.
        if not is_perfect_match:
            return False

        # NOTE: 2. The dtypes of inputs and attributes should be in the
        # type constraints of the OpSchema. If they are not, we know the function is not
        # eligible to be a perfect match, but can be a nearest match candidate.
        for schema_input, torch_input in zip(self.op_schema.inputs, function_inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if not allowed_types.intersection(torch_input_compatible_types) and not any(
                fx_type_utils.is_optional_onnx_dtype_str(onnx_type_str)
                for onnx_type_str in allowed_types
            ):
                # If torch_input_compatible_types isn't in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are not compatible
                logger.info(
                    "Actual %s vs\nExpected %s",
                    torch_input_compatible_types,
                    allowed_types,
                )
                is_perfect_match = False

        for attribute_name, attribute in function_attributes.items():
            if not self._match_onnx_attribute_type(attribute_name, attribute):
                # If the attribute type of the OpSchema and the attribute type don't match,
                # we know the function and the input are not compatible
                logger.info(
                    "Actual %s vs\nExpected %s",
                    type(attribute),
                    self.attributes[attribute_name].type,
                )
                is_perfect_match = False

        # NOTE: This is still a candidate for nearest match, as it only mismatches attributes on dtype.
        self._record_matching_score(function_inputs, function_attributes)
        logger.info("match score: %d", self.match_score)
        return is_perfect_match

    def _match_onnx_attribute_type(
        self,
        attribute_name: str,
        attribute: fx_type_utils.Argument | onnxscript_graph_building.TorchScriptTensor,
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

    def _record_matching_score(
        self,
        inputs: Sequence[
            fx_type_utils.TensorLike | str | int | float | bool | list | complex | None
        ],
        attributes: dict[str, fx_type_utils.Argument],
    ):
        """Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        Only the functions which have the same number of inputs and attributes as the
        OpSchema are eligible to be a nearest match candidate. Thus, we don't need to
        check the length of inputs and attributes here, and only check the types of
        inputs and attributes.

        How the matchsing score is calculated:
            score += 1 if one input/attribute type is in the type constraints.

        Limitations:
            None/NoeType/[] could result in zero matches, and the same score of overloads.

        Args:
            inputs: The input arguments.
            attributes: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        self._matching_score = 0
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
            attribute = attributes[attribute_name]
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(
                type(attribute)
            )
            if attribute_onnx_type != attribute_proto.type:
                # If the attribute type of the OpSchema and the attribute type don't match,
                # we know the function and the input are not compatible
                self._matching_score -= 1

    # NOTE: Referenced from onnxscript internal function.
    # Importing this function makes the code less robust, as it is not a public API.

    def _separate_input_attributes_from_arguments(
        self,
        param_schemas: Sequence[onnxscript.values.ParamSchema],
        args: Sequence[
            fx_type_utils.TensorLike | str | int | float | bool | list | complex | None
        ],
        kwargs: dict[str, fx_type_utils.Argument],
        fill_defaults: bool = True,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Separate Python args and kwargs into ONNX inputs and attributes.

        Extra_kwargs are ignored if their values are None. For example, if the
        OpSchema has an attribute "rounding_mode" and the caller provides
        "rounding_mode=None", the attribute "rounding_mode" will not be included
        in the returned attributes when the OnnxFunction signature doesn't have
        "rounding_mode" as an attribute.

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

        import onnx

        onnx_inputs: list[Any] = []
        onnx_attributes: dict[str, Any] = {}
        # NOTE: We need to copy kwargs because we will mutate it
        copy_kwargs = kwargs.copy()
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
            elif param.name in copy_kwargs:
                if param.is_input:
                    # Move the input from kwargs to inputs
                    onnx_inputs.append(copy_kwargs[param.name])
                    copy_kwargs.pop(param.name)
                else:
                    onnx_attributes[param.name] = copy_kwargs[param.name]
            elif (
                param.is_attribute
                and self.attributes[param.name].default_value.type
                != onnx.AttributeProto.UNDEFINED  # type: ignore[attr-defined]
            ):
                # User did not provide the attribute
                if fill_defaults:
                    onnx_attributes[param.name] = param.default
            # optional input
            elif param.is_input:
                if fill_defaults:
                    onnx_inputs.append(None)

        # NOTE: Pick up extra kwargs if it's not None. None is not expected
        # as an attribute value in torchlib.
        for k, v in copy_kwargs.items():
            if k not in onnx_attributes and v is not None:
                onnx_attributes[k] = v
        return onnx_inputs, onnx_attributes


def _is_arg_with_complex_dtype(arg: fx_type_utils.Argument) -> bool:
    """Check if the node has complex dtype recursively."""
    if (
        isinstance(arg, torch.fx.Node)
        and "val" in arg.meta
        and isinstance(arg.meta["val"], torch.Tensor)
        and torch.is_complex(arg.meta["val"])
    ):
        return True
    elif isinstance(arg, list):
        for item in arg:
            return _is_arg_with_complex_dtype(item)
    return False


def _find_onnx_data_type(
    torch_input: fx_type_utils.TensorLike
    | str
    | int
    | float
    | bool
    | list
    | tuple
    | complex
    | None,
) -> set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    if (
        isinstance(torch_input, fx_type_utils.TensorLike)
        and torch_input.dtype is not None
    ):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(torch_input.dtype)
    if isinstance(torch_input, (int, float, bool, str, complex)):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(type(torch_input))
    if isinstance(torch_input, (list, tuple)) and torch_input:  # [Tensor, Tensor]
        the_first_non_none_item = next(
            (item for item in torch_input if item is not None), None
        )
        set_dtype = _find_onnx_data_type(the_first_non_none_item)
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
