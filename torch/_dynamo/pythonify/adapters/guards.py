"""
GuardAdapter for translating Dynamo guards to GuardCheckNode IR nodes.

This module provides functionality to translate the various guard types used
by Dynamo's guard system into the structured GuardCheckNode IR representation.
The translated IR nodes can then be used by code generation backends to produce
either runtime guard checking code (gen_binary) or explicit Python assertions
(gen_python).

Dynamo's guard system includes many guard types including:
- TENSOR_MATCH: Comprehensive tensor property guards (dtype, device, shape, stride)
- TYPE_MATCH: Object type identity guards
- EQUALS_MATCH: Value equality guards
- DICT_KEYS_MATCH: Dictionary keys guards
- Shape guards from symbolic shape expressions

The GuardAdapter translates these into the simplified GuardCheckNode IR nodes
with types: SHAPE, DTYPE, DEVICE, VALUE, TENSOR_MATCH, IDENTITY, TYPE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, TYPE_CHECKING

from ..ir import GuardCheckNode, GuardType


if TYPE_CHECKING:
    from torch._guards import Guard


class DynamoGuardType(Enum):
    """
    Enum representing the different guard types that Dynamo can generate.

    These correspond to the methods on GuardBuilder in torch/_dynamo/guards.py.
    Not all guard types are directly translatable to simple IR nodes; some
    (like TENSOR_MATCH) expand to multiple checks.
    """

    TENSOR_MATCH = auto()
    TYPE_MATCH = auto()
    EQUALS_MATCH = auto()
    ID_MATCH = auto()
    DICT_VERSION = auto()
    DICT_CONTAINS = auto()
    DICT_KEYS_MATCH = auto()
    SET_CONTAINS = auto()
    LAMBDA = auto()
    SHAPE_ENV = auto()
    BUILTIN_MATCH = auto()
    DATA_PTR_MATCH = auto()
    DYNAMIC_INDICES = auto()
    NO_HASATTR = auto()
    HASATTR = auto()
    NN_MODULE = auto()
    GLOBAL_STATE = auto()
    DEFAULT_DEVICE = auto()
    NOT_NONE = auto()
    CONSTANT_MATCH = auto()
    CLOSURE_MATCH = auto()
    PYTHON_LAMBDA = auto()
    ALIASING = auto()
    UNKNOWN = auto()


@dataclass
class GuardInfo:
    """
    Intermediate representation of guard information extracted from Dynamo.

    This captures the essential information needed to generate a GuardCheckNode,
    extracted from Dynamo's more complex guard representation.

    Attributes:
        guard_type: The Dynamo guard type
        source_name: The source expression being guarded (e.g., "L['x']")
        target_name: Simplified name for the guarded variable (e.g., "x")
        code_parts: The code expressions that implement the guard
        verbose_code_parts: Human-readable versions with context
        expected_value: The expected value for equality-based guards
        metadata: Additional guard-specific metadata
    """

    guard_type: DynamoGuardType
    source_name: str
    target_name: str
    code_parts: list[str] = field(default_factory=list)
    verbose_code_parts: list[str] = field(default_factory=list)
    expected_value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorGuardInfo:
    """
    Specialized guard info for tensor-related guards.

    Tensor guards are complex and check multiple properties. This class
    captures all the tensor properties that need to be guarded.

    Attributes:
        tensor_name: The name/source of the tensor being guarded
        dtype: Expected dtype (e.g., torch.float32)
        device_type: Expected device type (e.g., 'cuda', 'cpu')
        device_index: Expected device index (e.g., 0 for cuda:0)
        requires_grad: Expected requires_grad value
        shape: Expected shape as list of ints (None for dynamic dims)
        stride: Expected stride as list of ints (None for dynamic dims)
        dispatch_key: Expected dispatch key set
        pytype: Expected Python type of the tensor
        dynamic_dims: Set of dimension indices that are dynamic (no guard generated)
        shape_constraints: Dict mapping dimension index to (min, max) constraints
    """

    tensor_name: str
    dtype: Any = None
    device_type: Optional[str] = None
    device_index: Optional[int] = None
    requires_grad: Optional[bool] = None
    shape: Optional[list[Optional[int]]] = None
    stride: Optional[list[Optional[int]]] = None
    dispatch_key: Any = None
    pytype: Optional[type] = None
    dynamic_dims: set[int] = field(default_factory=set)
    shape_constraints: dict[int, tuple[Optional[int], Optional[int]]] = field(
        default_factory=dict
    )


class GuardAdapter:
    """
    Adapter for translating Dynamo guards to GuardCheckNode IR nodes.

    This adapter takes guard information from Dynamo's guard system and
    produces the corresponding GuardCheckNode IR nodes that can be processed
    by the pythonify code generation backends.

    The adapter handles several forms of input:
    1. Guard dictionaries (simplified format from CompilationArtifacts)
    2. GuardInfo objects (intermediate representation)
    3. TensorGuardInfo objects (for tensor-specific guards)

    Usage:
        adapter = GuardAdapter()

        # From a simple guard dictionary
        nodes = adapter.translate_guard_dict({
            "type": "shape",
            "target": "x",
            "condition": "x.shape[0] == 3",
            "expected_value": 3,
            "dimension": 0,
        })

        # From tensor guard info
        tensor_info = TensorGuardInfo(
            tensor_name="x",
            dtype=torch.float32,
            device_type="cuda",
            shape=[3, 4],
        )
        nodes = adapter.translate_tensor_guard(tensor_info)
    """

    def __init__(self) -> None:
        """Initialize the GuardAdapter."""
        pass

    def translate_guard_dict(self, guard_dict: dict[str, Any]) -> list[GuardCheckNode]:
        """
        Translate a guard dictionary to GuardCheckNode IR nodes.

        This is the primary entry point for translating guards from the
        CompilationArtifacts format used by RuntimeWrapperPipeline.

        Args:
            guard_dict: Dictionary with guard information. Expected keys:
                - type: Guard type string ("shape", "dtype", "device", "value", etc.)
                - target: Name of the variable being guarded
                - condition: The condition expression
                - expected_value: Expected value for the guard
                - dimension: (optional) Dimension index for shape guards
                - error_message: (optional) Custom error message

        Returns:
            List of GuardCheckNode objects representing this guard
        """
        guard_type_str = guard_dict.get("type", "").lower()
        target = guard_dict.get("target", "")
        condition = guard_dict.get("condition", "")
        expected_value = guard_dict.get("expected_value")
        dimension = guard_dict.get("dimension")
        error_message = guard_dict.get("error_message", "")

        guard_type = self._map_type_string_to_guard_type(guard_type_str)

        node = GuardCheckNode(
            guard_type=guard_type,
            target_name=target,
            condition=condition,
            expected_value=expected_value,
            dimension=dimension,
            error_message=error_message or self._generate_error_message(
                guard_type, target, expected_value, dimension
            ),
        )

        return [node]

    def translate_guard_info(self, info: GuardInfo) -> list[GuardCheckNode]:
        """
        Translate a GuardInfo object to GuardCheckNode IR nodes.

        Args:
            info: GuardInfo object with extracted guard information

        Returns:
            List of GuardCheckNode objects
        """
        nodes = []

        if info.guard_type == DynamoGuardType.TYPE_MATCH:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.TYPE,
                target_name=info.target_name,
                condition=info.code_parts[0] if info.code_parts else "",
                expected_value=info.expected_value,
                error_message=self._generate_error_message(
                    GuardType.TYPE, info.target_name, info.expected_value
                ),
            ))

        elif info.guard_type == DynamoGuardType.EQUALS_MATCH:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.VALUE,
                target_name=info.target_name,
                condition=info.code_parts[0] if info.code_parts else "",
                expected_value=info.expected_value,
                error_message=self._generate_error_message(
                    GuardType.VALUE, info.target_name, info.expected_value
                ),
            ))

        elif info.guard_type == DynamoGuardType.ID_MATCH:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.IDENTITY,
                target_name=info.target_name,
                condition=info.code_parts[0] if info.code_parts else "",
                expected_value=info.expected_value,
                error_message=self._generate_error_message(
                    GuardType.IDENTITY, info.target_name, info.expected_value
                ),
            ))

        elif info.guard_type == DynamoGuardType.TENSOR_MATCH:
            tensor_info = self._extract_tensor_info_from_guard(info)
            nodes.extend(self.translate_tensor_guard(tensor_info))

        elif info.guard_type == DynamoGuardType.SHAPE_ENV:
            shape_nodes = self._translate_shape_env_guard(info)
            nodes.extend(shape_nodes)

        else:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.VALUE,
                target_name=info.target_name,
                condition=info.code_parts[0] if info.code_parts else "",
                expected_value=info.expected_value,
                error_message=f"Guard check failed for {info.target_name}",
            ))

        return nodes

    def translate_tensor_guard(self, tensor_info: TensorGuardInfo) -> list[GuardCheckNode]:
        """
        Translate tensor guard information to multiple GuardCheckNode IR nodes.

        A single TENSOR_MATCH guard in Dynamo expands to multiple checks:
        dtype, device, requires_grad, shape (per dimension), and stride.

        For dynamic dimensions (either marked via dynamic_dims set or having None
        shape values), the generated GuardCheckNode will have is_dynamic=True,
        which means the code generator will emit a comment instead of an assertion.

        If shape_constraints are specified for a dynamic dimension, min_value and
        max_value will be set on the GuardCheckNode to generate bound checks.

        Args:
            tensor_info: TensorGuardInfo with tensor properties to check

        Returns:
            List of GuardCheckNode objects for all tensor property checks
        """
        nodes = []
        name = tensor_info.tensor_name

        if tensor_info.dtype is not None:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.DTYPE,
                target_name=name,
                condition=f"{name}.dtype == {tensor_info.dtype}",
                expected_value=tensor_info.dtype,
                error_message=f"Expected {name}.dtype to be {tensor_info.dtype}",
            ))

        if tensor_info.device_type is not None:
            device_str = tensor_info.device_type
            if tensor_info.device_index is not None:
                device_str = f"{tensor_info.device_type}:{tensor_info.device_index}"
            nodes.append(GuardCheckNode(
                guard_type=GuardType.DEVICE,
                target_name=name,
                condition=f"str({name}.device) == '{device_str}'",
                expected_value=device_str,
                error_message=f"Expected {name}.device to be {device_str}",
            ))

        if tensor_info.requires_grad is not None:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.VALUE,
                target_name=name,
                condition=f"{name}.requires_grad == {tensor_info.requires_grad}",
                expected_value=tensor_info.requires_grad,
                error_message=f"Expected {name}.requires_grad to be {tensor_info.requires_grad}",
            ))

        if tensor_info.shape is not None:
            for dim, size in enumerate(tensor_info.shape):
                is_dynamic = (
                    dim in tensor_info.dynamic_dims or size is None
                )

                if is_dynamic:
                    min_val, max_val = tensor_info.shape_constraints.get(dim, (None, None))
                    nodes.append(GuardCheckNode(
                        guard_type=GuardType.SHAPE,
                        target_name=name,
                        condition=f"{name}.shape[{dim}] is dynamic",
                        expected_value=None,
                        dimension=dim,
                        error_message=f"{name}.shape[{dim}] is dynamic",
                        is_dynamic=True,
                        min_value=min_val,
                        max_value=max_val,
                    ))
                elif size is not None:
                    nodes.append(GuardCheckNode(
                        guard_type=GuardType.SHAPE,
                        target_name=name,
                        condition=f"{name}.shape[{dim}] == {size}",
                        expected_value=size,
                        dimension=dim,
                        error_message=f"Expected {name}.shape[{dim}] to be {size}",
                    ))

        if tensor_info.pytype is not None:
            nodes.append(GuardCheckNode(
                guard_type=GuardType.TYPE,
                target_name=name,
                condition=f"type({name}) is {tensor_info.pytype.__name__}",
                expected_value=tensor_info.pytype,
                error_message=f"Expected type({name}) to be {tensor_info.pytype.__name__}",
            ))

        return nodes

    def translate_shape_expression(
        self,
        target_name: str,
        dimension: int,
        expression: str,
        expected_value: Any,
    ) -> GuardCheckNode:
        """
        Translate a symbolic shape expression to a GuardCheckNode.

        This handles shape guards from Dynamo's symbolic shape system,
        where shapes may be expressed as symbolic expressions.

        Args:
            target_name: Name of the tensor
            dimension: Dimension index being guarded
            expression: The shape guard expression
            expected_value: Expected value or symbolic expression

        Returns:
            GuardCheckNode for this shape guard
        """
        return GuardCheckNode(
            guard_type=GuardType.SHAPE,
            target_name=target_name,
            condition=expression,
            expected_value=expected_value,
            dimension=dimension,
            error_message=f"Shape guard failed: {expression}",
        )

    def parse_guard_code_parts(
        self,
        code_parts: list[str],
        source_name: str,
    ) -> list[GuardCheckNode]:
        """
        Parse guard code parts to extract GuardCheckNode information.

        This parses the code expressions generated by Dynamo's guard system
        and extracts the relevant guard information.

        Args:
            code_parts: List of guard code expressions
            source_name: The source expression being guarded

        Returns:
            List of GuardCheckNode objects
        """
        nodes = []
        target_name = self._simplify_source_name(source_name)

        for code in code_parts:
            node = self._parse_single_code_part(code, target_name)
            if node is not None:
                nodes.append(node)

        return nodes

    def _map_type_string_to_guard_type(self, type_str: str) -> GuardType:
        """Map a guard type string to a GuardType enum value."""
        mapping = {
            "shape": GuardType.SHAPE,
            "dtype": GuardType.DTYPE,
            "device": GuardType.DEVICE,
            "value": GuardType.VALUE,
            "tensor_match": GuardType.TENSOR_MATCH,
            "identity": GuardType.IDENTITY,
            "type": GuardType.TYPE,
            "id_match": GuardType.IDENTITY,
            "type_match": GuardType.TYPE,
            "equals_match": GuardType.VALUE,
        }
        return mapping.get(type_str, GuardType.VALUE)

    def _generate_error_message(
        self,
        guard_type: GuardType,
        target_name: str,
        expected_value: Any,
        dimension: Optional[int] = None,
    ) -> str:
        """Generate a human-readable error message for a guard failure."""
        if guard_type == GuardType.SHAPE:
            if dimension is not None:
                return f"Expected {target_name}.shape[{dimension}] to be {expected_value}"
            return f"Shape mismatch for {target_name}"
        elif guard_type == GuardType.DTYPE:
            return f"Expected {target_name}.dtype to be {expected_value}"
        elif guard_type == GuardType.DEVICE:
            return f"Expected {target_name}.device to be {expected_value}"
        elif guard_type == GuardType.TYPE:
            return f"Expected type({target_name}) to be {expected_value}"
        elif guard_type == GuardType.IDENTITY:
            return f"Object identity mismatch for {target_name}"
        else:
            return f"Guard check failed: {target_name} != {expected_value}"

    def _simplify_source_name(self, source_name: str) -> str:
        """
        Simplify a Dynamo source name to a simple variable name.

        Dynamo source names can be complex expressions like "L['x']" or
        "G['module'].weight". This extracts a simpler name for display.
        """
        match = re.search(r"\['(\w+)'\]", source_name)
        if match:
            return match.group(1)

        match = re.search(r"\.(\w+)$", source_name)
        if match:
            return match.group(1)

        return source_name

    def _extract_tensor_info_from_guard(self, info: GuardInfo) -> TensorGuardInfo:
        """Extract TensorGuardInfo from a GuardInfo object."""
        metadata = info.metadata
        return TensorGuardInfo(
            tensor_name=info.target_name,
            dtype=metadata.get("dtype"),
            device_type=metadata.get("device_type"),
            device_index=metadata.get("device_index"),
            requires_grad=metadata.get("requires_grad"),
            shape=metadata.get("shape"),
            stride=metadata.get("stride"),
            dispatch_key=metadata.get("dispatch_key"),
            pytype=metadata.get("pytype"),
        )

    def _translate_shape_env_guard(self, info: GuardInfo) -> list[GuardCheckNode]:
        """Translate shape environment guards to GuardCheckNode objects."""
        nodes = []

        for code in info.code_parts:
            shape_match = re.search(
                r"(\w+)\.shape\[(\d+)\]\s*(==|<=|>=|<|>)\s*(\d+)",
                code,
            )
            if shape_match:
                target = shape_match.group(1)
                dim = int(shape_match.group(2))
                op = shape_match.group(3)
                value = int(shape_match.group(4))

                nodes.append(GuardCheckNode(
                    guard_type=GuardType.SHAPE,
                    target_name=target,
                    condition=code,
                    expected_value=value,
                    dimension=dim,
                    error_message=f"Shape guard failed: {code}",
                ))

        return nodes

    def _parse_single_code_part(
        self,
        code: str,
        target_name: str,
    ) -> Optional[GuardCheckNode]:
        """Parse a single code part into a GuardCheckNode if possible."""
        if ".shape[" in code:
            match = re.search(r"\.shape\[(\d+)\]\s*==\s*(\d+)", code)
            if match:
                dim = int(match.group(1))
                value = int(match.group(2))
                return GuardCheckNode(
                    guard_type=GuardType.SHAPE,
                    target_name=target_name,
                    condition=code,
                    expected_value=value,
                    dimension=dim,
                    error_message=f"Expected {target_name}.shape[{dim}] to be {value}",
                )

        if ".dtype ==" in code:
            match = re.search(r"\.dtype\s*==\s*(\S+)", code)
            if match:
                dtype_str = match.group(1)
                return GuardCheckNode(
                    guard_type=GuardType.DTYPE,
                    target_name=target_name,
                    condition=code,
                    expected_value=dtype_str,
                    error_message=f"Expected {target_name}.dtype to be {dtype_str}",
                )

        if ".device" in code:
            return GuardCheckNode(
                guard_type=GuardType.DEVICE,
                target_name=target_name,
                condition=code,
                expected_value=None,
                error_message=f"Device check failed for {target_name}",
            )

        if "___check_type_id" in code or "type(" in code:
            return GuardCheckNode(
                guard_type=GuardType.TYPE,
                target_name=target_name,
                condition=code,
                expected_value=None,
                error_message=f"Type check failed for {target_name}",
            )

        if "___check_obj_id" in code or "id(" in code:
            return GuardCheckNode(
                guard_type=GuardType.IDENTITY,
                target_name=target_name,
                condition=code,
                expected_value=None,
                error_message=f"Identity check failed for {target_name}",
            )

        return GuardCheckNode(
            guard_type=GuardType.VALUE,
            target_name=target_name,
            condition=code,
            expected_value=None,
            error_message=f"Guard check failed: {code}",
        )


def translate_guards(
    guards: list[dict[str, Any]],
) -> list[GuardCheckNode]:
    """
    Convenience function to translate a list of guard dictionaries.

    This is the main entry point for translating guards from
    CompilationArtifacts to GuardCheckNode IR nodes.

    Args:
        guards: List of guard dictionaries

    Returns:
        List of GuardCheckNode objects
    """
    adapter = GuardAdapter()
    nodes = []
    for guard_dict in guards:
        nodes.extend(adapter.translate_guard_dict(guard_dict))
    return nodes


def create_dynamic_shape_guard(
    tensor_name: str,
    dimension: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> GuardCheckNode:
    """
    Create a dynamic shape guard for a tensor dimension.

    Dynamic shape guards indicate that a dimension can vary and should not
    have a strict equality assertion. Instead, they may have optional min/max
    bounds that are checked.

    This is useful when torch.compile is called with dynamic=True or when
    specific dimensions are marked as dynamic.

    Args:
        tensor_name: Name of the tensor variable
        dimension: The dimension index that is dynamic
        min_value: Optional minimum value for the dimension (checked if provided)
        max_value: Optional maximum value for the dimension (checked if provided)

    Returns:
        A GuardCheckNode with is_dynamic=True

    Example:
        guard = create_dynamic_shape_guard("x", 0, min_value=1, max_value=128)

        Generates code like:
            # DYNAMIC: x.shape[0] can vary (no assertion)
            assert x.shape[0] >= 1, "Dynamic dimension x.shape[0] must be >= 1"
            assert x.shape[0] <= 128, "Dynamic dimension x.shape[0] must be <= 128"
    """
    return GuardCheckNode(
        guard_type=GuardType.SHAPE,
        target_name=tensor_name,
        condition=f"{tensor_name}.shape[{dimension}] is dynamic",
        expected_value=None,
        dimension=dimension,
        error_message=f"{tensor_name}.shape[{dimension}] is dynamic",
        is_dynamic=True,
        min_value=min_value,
        max_value=max_value,
    )
