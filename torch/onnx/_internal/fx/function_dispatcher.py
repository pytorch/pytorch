"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import operator
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
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
from torch.onnx import _constants, _type_utils
from torch.onnx._internal import _beartype

from torch.onnx._internal.fx import registration

if TYPE_CHECKING:
    import onnx.defs  # type: ignore[import]
    import onnxscript  # type: ignore[import]

_SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE: Dict[
    Union[Callable[..., Any], str], torch._ops.OpOverload
] = {
    operator.mul: torch.ops.aten.mul.default,  # type: ignore[has-type]
    operator.add: torch.ops.aten.add.default,  # type: ignore[has-type]
    operator.pow: torch.ops.aten.pow.int,  # type: ignore[has-type]
    operator.sub: torch.ops.aten.sub.default,  # type: ignore[has-type]
}


class OnnxDispatcher:
    """
    The OnnxDispatcher class finds the best matched function for a given aten operation
    in the FX exporter. It uses the torch.ops name to find the function. If not found,
    it falls back to default. Then, it finds the best match among all overloaded
    functions. If the types match, it selects the function. Otherwise, it finds the
    nearest upcast/downcast: Float to Float / Int to Int, Int to Float.

    Steps for overloaded function dispatch:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists.
        b. If not found, check ATen overload=default.

    2. Find the best match among all overloaded functions:
        a. If the types match, select the function.
    """

    def __init__(self, registry: registration.OnnxRegistry, opset_version: int):
        """Initialize the Dispatcher.

        Args:
            registry: The registration registry.
            opset_version: The model opset version.
        """
        self._registry = registry
        self._opset_version = opset_version

    @property
    def opset_version(self) -> int:
        """Get the model opset version."""
        return self._opset_version

    @property
    def registry(self) -> registration.OnnxRegistry:
        """Get the registration registry."""
        return self._registry

    @_beartype.beartype
    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence,
        onnx_kwargs: Mapping,
    ) -> Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]:
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
        aten_name = self._get_aten_name(node)
        function_overloads = self._get_function_overloads(node, aten_name)

        overload_match_ranking: Dict[
            Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction], int
        ] = {}
        # TODO: Cache the OpSchemaWrapper so we don't need to run the init logic everytime
        for overload_func in function_overloads:
            function_opschema = OpSchemaWrapper(overload_func.op_schema)
            if function_opschema.match_inputs(onnx_args, onnx_kwargs):
                # If the perfect match is found, return the function
                return overload_func
            # put the function into candidate list with its match score
            overload_match_ranking[overload_func] = function_opschema.match_score
        # TODO(titaiwang): Add diagnostic message saying not perfect match get the best match
        return max(overload_match_ranking, key=overload_match_ranking.get)  # type: ignore[arg-type]

    def dispatch_opset_version(
        self, target: int, registered_opsets: Collection[int]
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

    def _get_aten_name(self, node: torch.fx.Node) -> str:
        """Get the aten name from the target."""
        if node.target == operator.getitem:
            return "aten::getitem"
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            # aten::sym_size is the only OverloadPacket that we support.
            # schema: aten::sym_size(Tensor self, int dim) -> Tensor
            if node.target != torch.ops.aten.sym_size:
                raise ValueError(
                    f"Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!"
                )
            # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
            # overloadpacket for some reasons.
            # https://github.com/pytorch/pytorch/issues/97201
            aten_op_default = node.target.default
            return aten_op_default.name()  # type: ignore[attr-defined]
        if node.target in _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE:
            # Make sure it's symint/symfloat consuming builtin ops.
            for node_arg in node.args:
                if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                    isinstance(node_arg, torch.fx.Node)
                    and not isinstance(
                        node_arg.meta["val"], (torch.SymInt, torch.SymFloat)
                    )
                ):
                    raise ValueError(
                        f"Unsupported node arg: {node_arg} with builtin function: {node.target},"
                        " only int/float/SymInt/SymFloat is supported with built-in ops!"
                    )
            aten_op = _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE[node.target]
            return aten_op.name()
        if isinstance(node.target, torch._ops.OpOverload):
            return node.target.name()

        raise RuntimeError(f"Unknown call_function target: {node.target}")

    def _get_function_overloads(
        self, node: torch.fx.Node, aten_name: str
    ) -> List[Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]]:
        """Get the function overloads from the registry."""
        function_group = None

        if self.registry.is_registered_op(aten_name, self.opset_version):
            function_group = self.registry.get_function_group(aten_name)

        # Fall back to overloadpacket name: eg: aten.add.Tensor -> aten::add
        elif hasattr(node.target, "overloadpacket") and self.registry.is_registered_op(
            node.target.overloadpacket._qualified_op_name,  # type: ignore[union-attr]
            self.opset_version,
        ):
            function_group = self.registry.get_function_group(
                node.target.overloadpacket._qualified_op_name  # type: ignore[union-attr]
            )

        if function_group is not None:
            # TODO(titaiwang): dispatch opset version.
            dispatched_version = self.dispatch_opset_version(
                self.opset_version, function_group.support_opset()
            )
            if dispatched_version is not None:
                function_overloads = function_group.get(dispatched_version)
                if function_overloads is not None:
                    return function_overloads

        raise RuntimeError(
            f"aten name: {aten_name} is not registered in the ONNX registry!"
        )


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class _TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


class OpSchemaWrapper:
    """
    The OpSchemaWrapper class is a wrapper for ONNX OpSchema.

    It provides methods to check for input compatibility based on the OpSchema.

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
        self.attributes = set(op_schema.attributes)
        self._matching_score: int = 0

    @property
    def match_score(self) -> int:
        """The matching score of the OpSchemaWrapper.

        Returns:
            The matching score of the OpSchemaWrapper.
        """
        return self._matching_score

    def match_inputs(
        self, args: Sequence[Union[_TensorLike, str, int, float, bool]], kwargs
    ) -> bool:
        """Check if the inputs match the OpSchema requirements.

        Args:
            args: The input arguments.
            kwargs: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """

        # TODO: Refine the logic for it to be more robust
        # TODO: Handle attributes
        for schema_input, torch_input in zip(self.schema.inputs, args):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types is in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are not compatible
                self._matching_score += 1
        for attr in kwargs:
            if attr in self.attributes:
                self._matching_score += 1
        return self._matching_score == (
            len(self.schema.inputs) + max(len(self.attributes), len(kwargs))
        )


@_beartype.beartype
def _find_onnx_data_type(
    torch_input: Optional[Union[_TensorLike, str, int, float, bool, list, tuple]]
) -> Set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    if isinstance(torch_input, _TensorLike) and torch_input.dtype is not None:
        return _type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
            torch_input.dtype
        ]
    if isinstance(torch_input, (int, float, bool, str)):
        return _type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
            type(torch_input)
        ]
    if isinstance(torch_input, (list, tuple)) and torch_input:
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
        return set().union(
            *_type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS.values()
        )

    raise RuntimeError(f"Unknown input type from input: {torch_input}")
