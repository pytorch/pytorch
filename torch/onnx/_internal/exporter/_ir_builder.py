r"""Direct ONNX IR construction for plain-callable translations.

Only Python constants are converted implicitly. Tensor casts, symbolic shape
construction, sequences, and control flow remain explicit ONNX operations.
This module does not depend on ONNX Script evaluation or annotations.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, TYPE_CHECKING

import onnx

import onnx_ir as ir


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


_CURRENT: contextvars.ContextVar[GraphBuilder | None] = contextvars.ContextVar(
    "onnx_ir_graph_builder", default=None
)


def current_builder() -> GraphBuilder:
    r"""Return the builder installed by ``GraphBuilder.activate()``."""
    builder = _CURRENT.get()
    if builder is None:
        raise RuntimeError(
            "No ONNX IR builder is active. Run the translation inside "
            "'with builder.activate():' or pass the builder explicitly."
        )
    return builder


class GraphBuilder:
    r"""Append native nodes to one graph without changing existing exports.

    Always returns tuples of plain ir.Value objects. A builder has graph-local
    constants and scoped activation; no symbolic Python operators are installed.
    Model assembly and optimization belong to the caller.

    Example::

        graph = ir.Graph([], [], nodes=[], name="translation")
        builder = GraphBuilder(graph)
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT))
        with builder.activate():
            (result,) = current_builder().call("Add", x, 1)
        graph.inputs.append(x)
        graph.outputs.append(result)
    """

    def __init__(self, graph: ir.Graph, *, opset: int = 18) -> None:
        existing = graph.opset_imports.get("")
        if existing is not None and existing != opset:
            raise ValueError(
                f"Graph imports ONNX opset {existing}, not {opset}. "
                "Construct the builder with the graph's opset."
            )
        self.graph = graph
        self.opset = opset
        self.functions: dict[ir.OperatorIdentifier, ir.Function] = {}
        self._constants: dict[tuple[ir.DataType, tuple[int, ...], bytes], ir.Value] = {}

    @contextlib.contextmanager
    def activate(self) -> Iterator[GraphBuilder]:
        r"""Install this builder, restoring the previous scope after exceptions."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def constant(self, value: Any, *, dtype: ir.DataType) -> ir.Value:
        r"""Create a typed tensor constant; empty tensors require an explicit dtype."""
        try:
            tensor = ir.tensor(value, dtype=dtype)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot construct an ONNX constant with dtype {dtype}. "
                "Supply rectangular Python data representable by that dtype. "
                "Build symbolic shape lists explicitly with Reshape/Concat."
            ) from error
        array = tensor.numpy()
        # Object/string arrays contain pointers rather than stable value bytes.
        cacheable = array.dtype.kind not in ("O", "U", "S")
        key = (dtype, tuple(array.shape), array.tobytes())
        if cacheable and key in self._constants:
            return self._constants[key]
        result = self.node(
            "Constant",
            attributes={"value": tensor},
            output_types=[ir.TensorType(dtype)],
        )[0]
        result.shape = tensor.shape
        result.const_value = tensor
        if cacheable:
            self._constants[key] = result
        return result

    def node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None] = (),
        *,
        attributes: Mapping[str, Any] | None = None,
        output_types: Sequence[ir.TypeProtocol | None],
        domain: str = "",
        version: int | None = None,
        overload: str = "",
    ) -> tuple[ir.Value, ...]:
        r"""Append a raw node with explicit types, including custom/control-flow nodes.

        Subgraphs are explicit ir.Graph attributes, constructed with their own
        builder. Captures, graph outputs, function dependencies, and names avoiding
        shadowing of ancestor values are the caller's responsibility.
        """
        if version is None:
            if domain:
                raise ValueError(
                    f"Missing opset version for {domain}::{op_type}. Pass version=."
                )
            version = self.opset
        existing = self.graph.opset_imports.get(domain)
        if existing is not None and existing != version:
            raise ValueError(
                f"Graph imports {domain!r} version {existing}, but {op_type} "
                f"requests {version}. Use one opset version per domain."
            )
        if any(
            value is not None and not isinstance(value, ir.Value) for value in inputs
        ):
            raise TypeError(
                f"{domain}::{op_type} raw inputs must be ir.Value or None. "
                "Use constant(..., dtype=...) or call() for Python constants."
            )
        trimmed = list(inputs)
        while trimmed and trimmed[-1] is None:
            trimmed.pop()
        outputs = [ir.Value(type=type_) for type_ in output_types]
        node = ir.Node(
            domain,
            op_type,
            inputs=trimmed,
            attributes=ir.convenience.convert_attributes(
                {
                    key: value
                    for key, value in (attributes or {}).items()
                    if value is not None
                }
            ),
            outputs=outputs,
            version=version,
            overload=overload,
        )
        self.graph.append(node)
        self.graph.opset_imports[domain] = version
        return tuple(outputs)

    def call(
        self,
        op_type: str,
        *inputs: Any,
        domain: str = "",
        version: int | None = None,
        _num_outputs: int | None = None,
        _output_types: Sequence[ir.TypeProtocol | None] | None = None,
        **attributes: Any,
    ) -> tuple[ir.Value, ...]:
        r"""Call an ONNX schema, converting constants but never casting tensors.

        Inputs are positional, attributes are keyword-only, variadics are expanded
        positionally, and optional holes use None. Ambiguous output arity requires
        _num_outputs or _output_types. Types not determined by bindings (e.g. Cast)
        may be supplied explicitly; this is not general type/shape inference.
        """
        num_outputs = _num_outputs
        output_types = _output_types
        if version is None:
            version = self.graph.opset_imports.get(domain)
        if version is None:
            if domain:
                raise ValueError(
                    f"Missing opset version for {domain}::{op_type}. Pass version=."
                )
            version = self.opset
        existing = self.graph.opset_imports.get(domain)
        if existing is not None and existing != version:
            raise ValueError(
                f"{domain}::{op_type} requests version {version}, but graph imports "
                f"{existing}. Use the graph's opset version."
            )
        try:
            schema = onnx.defs.get_schema(op_type, version, domain)
        except onnx.defs.SchemaError as error:
            raise ValueError(
                f"No ONNX schema for {domain}::{op_type} at version {version}. "
                "Correct the operator/opset or use node() with explicit output types."
            ) from error
        if not schema.min_input <= len(inputs) <= schema.max_input:
            raise ValueError(
                f"{op_type} expects {schema.min_input}..{schema.max_input} inputs, "
                f"got {len(inputs)}. Supply optional holes as None and expand variadics."
            )
        signature = ir.schemas.OpSignature.from_op_schema(schema)
        parameters = [
            p for p in signature.params if isinstance(p, ir.schemas.Parameter)
        ]
        bindings: dict[ir.schemas.TypeConstraintParam, ir.TypeProtocol] = {}
        expanded = []
        for index, value in enumerate(inputs):
            param = parameters[min(index, len(parameters) - 1)]
            expanded.append((param, value))
            if value is None:
                if param.required:
                    raise ValueError(
                        f"{op_type} input {param.name!r} is required. Supply a typed value."
                    )
                continue
            if not isinstance(value, ir.Value):
                continue
            if value.type is None:
                raise ValueError(
                    f"{op_type} input {param.name!r} has unknown type. Set its IR type "
                    "before schema-guided construction, or use node() with explicit types."
                )
            constraint = param.type_constraint
            if value.type not in constraint.allowed_types:
                raise TypeError(
                    f"{op_type} input {param.name!r} has type {value.type}, expected "
                    f"{constraint.allowed_types}. Insert an explicit Cast."
                )
            if param.homogeneous:
                previous = bindings.get(constraint)
                if previous is not None and previous != value.type:
                    raise TypeError(
                        f"{op_type} input {param.name!r} has type {value.type}, "
                        f"but {constraint.name} is bound to {previous}. "
                        "Insert an explicit Cast; tensor promotion is not automatic."
                    )
                bindings[constraint] = value.type

        tensor_bindings = set(bindings)
        prepared: list[ir.Value | None | tuple[Any, ir.DataType]] = []
        for param, value in expanded:
            if value is None or isinstance(value, ir.Value):
                prepared.append(value)
                continue
            constraint = param.type_constraint
            type_ = (
                bindings.get(constraint)
                if param.homogeneous and constraint in tensor_bindings
                else None
            )
            if type_ is None and len(constraint.allowed_types) == 1:
                type_ = next(iter(constraint.allowed_types))
            if type_ is None:
                type_ = ir.TensorType(self._constant_dtype(value, op_type, param.name))
            if (
                not isinstance(type_, ir.TensorType)
                or type_ not in constraint.allowed_types
            ):
                raise TypeError(
                    f"{op_type} input {param.name!r} cannot convert a Python constant "
                    f"to {type_}. Build sequences and optionals explicitly with ONNX nodes."
                )
            if param.homogeneous:
                previous = bindings.get(constraint)
                if previous is not None and previous != type_:
                    raise TypeError(
                        f"{op_type} constant input {param.name!r} inferred {type_}, "
                        f"but {constraint.name} is bound to {previous}. "
                        "Supply explicitly typed constants for mixed Python dtypes."
                    )
                bindings[constraint] = type_
            prepared.append((value, type_.dtype))

        if output_types is not None:
            if num_outputs is not None and num_outputs != len(output_types):
                raise ValueError(
                    "_num_outputs disagrees with _output_types. Supply matching lengths."
                )
            num_outputs = len(output_types)
        if num_outputs is None:
            if schema.min_output != schema.max_output:
                raise ValueError(
                    f"{op_type} has optional/variadic outputs. "
                    "Pass _num_outputs= or _output_types= explicitly."
                )
            num_outputs = schema.min_output
        if not schema.min_output <= num_outputs <= schema.max_output:
            raise ValueError(
                f"{op_type} requires {schema.min_output}..{schema.max_output} outputs, "
                f"got {num_outputs}. Correct num_outputs or output_types."
            )
        inferred = []
        attribute_dtype = None
        if op_type == "Cast":
            attribute_dtype = ir.DataType(attributes["to"])
        elif op_type == "ConstantOfShape":
            attribute_dtype = (
                attributes["value"].dtype
                if attributes.get("value") is not None
                else ir.DataType.FLOAT
            )
        elif op_type == "Constant":
            if attributes.get("value") is not None:
                attribute_dtype = attributes["value"].dtype
            else:
                for suffix, dtype in (
                    ("int", ir.DataType.INT64),
                    ("float", ir.DataType.FLOAT),
                    ("string", ir.DataType.STRING),
                ):
                    if (
                        f"value_{suffix}" in attributes
                        or f"value_{suffix}s" in attributes
                    ):
                        attribute_dtype = dtype
                        break
        elif op_type in (
            "RandomNormalLike",
            "RandomUniformLike",
            "Bernoulli",
            "Multinomial",
        ):
            attribute_dtype = ir.DataType(
                attributes.get(
                    "dtype",
                    inputs[0].dtype
                    if isinstance(inputs[0], ir.Value)
                    else ir.DataType.FLOAT,
                )
            )
        elif op_type in ("BlackmanWindow", "HammingWindow", "HannWindow"):
            attribute_dtype = ir.DataType(
                attributes.get("output_datatype", ir.DataType.FLOAT)
            )
        for index in range(num_outputs):
            param = signature.outputs[min(index, len(signature.outputs) - 1)]
            constraint = param.type_constraint
            type_ = bindings.get(constraint) if param.homogeneous else None
            if type_ is None and len(constraint.allowed_types) == 1:
                type_ = next(iter(constraint.allowed_types))
            if type_ is None and attribute_dtype is not None:
                type_ = ir.TensorType(attribute_dtype)
            if output_types is not None:
                explicit = output_types[index]
                if explicit is not None:
                    if explicit not in constraint.allowed_types or (
                        type_ is not None and explicit != type_
                    ):
                        raise TypeError(
                            f"{op_type} output {index} type {explicit} conflicts with "
                            f"schema/binding {type_}. Correct _output_types."
                        )
                    type_ = explicit
            inferred.append(type_)
        for name in attributes:
            if name not in schema.attributes:
                raise ValueError(
                    f"{op_type} has no attribute {name!r}. Check its ONNX schema."
                )
        for name, attribute in schema.attributes.items():
            if attribute.required and attributes.get(name) is None:
                raise ValueError(
                    f"{op_type} requires attribute {name!r}. Supply its value."
                )
        typed_attributes = {
            name: ir.convenience.convert_attribute(
                name, value, ir.AttributeType(schema.attributes[name].type)
            )
            for name, value in attributes.items()
            if value is not None
        }
        resolved = [
            self.constant(value[0], dtype=value[1])
            if isinstance(value, tuple)
            else value
            for value in prepared
        ]
        return self.node(
            op_type,
            resolved,
            domain=domain,
            version=version,
            attributes=typed_attributes,
            output_types=inferred,
        )

    @staticmethod
    def _constant_dtype(value: Any, op_type: str, name: str) -> ir.DataType:
        if isinstance(value, bool):
            return ir.DataType.BOOL
        if isinstance(value, int):
            return ir.DataType.INT64
        if isinstance(value, float):
            return ir.DataType.FLOAT
        if isinstance(value, str):
            return ir.DataType.STRING
        if isinstance(value, (list, tuple)) and value:
            dtypes = {
                GraphBuilder._constant_dtype(item, op_type, name) for item in value
            }
            if len(dtypes) == 1:
                return dtypes.pop()
        raise TypeError(
            f"{op_type} input {name!r} has no unambiguous Python constant dtype. "
            "Use constant(value, dtype=...) for empty/mixed data. Construct symbolic "
            "shape lists explicitly with Reshape/Concat."
        )

    def call_function(
        self,
        function: ir.Function,
        *inputs: ir.Value,
        version: int,
        attributes: Mapping[str, Any] | None = None,
    ) -> tuple[ir.Value, ...]:
        r"""Call an explicit function; expose definitions for caller model assembly."""
        if len(inputs) != len(function.inputs):
            raise ValueError(
                f"Function {function.name} expects {len(function.inputs)} inputs, "
                f"got {len(inputs)}. Supply all function inputs."
            )
        for value, formal in zip(inputs, function.inputs):
            if formal.type is not None and value.type != formal.type:
                raise TypeError(
                    f"Function {function.name} input {formal.name} expects {formal.type}, "
                    f"got {value.type}. Set types or insert an explicit Cast."
                )
        supplied = attributes or {}
        for name in supplied:
            if name not in function.attributes:
                raise ValueError(
                    f"Function {function.name} has no attribute {name!r}. "
                    "Supply only attributes declared by the function."
                )
        typed_attributes = {}
        for name, formal in function.attributes.items():
            if supplied.get(name) is not None:
                typed_attributes[name] = ir.convenience.convert_attribute(
                    name, supplied[name], formal.type
                )
            elif formal.value is None:
                raise ValueError(
                    f"Function {function.name} requires attribute {name!r}. "
                    "Supply it in attributes=."
                )
        identifier = function.identifier()
        previous = self.functions.get(identifier)
        if previous is not None and previous is not function:
            raise ValueError(
                f"Function {identifier} is already defined. Reuse it or choose another name."
            )
        outputs = self.node(
            function.name,
            inputs,
            domain=function.domain,
            version=version,
            overload=function.overload,
            attributes=typed_attributes,
            output_types=[value.type for value in function.outputs],
        )
        self.functions[identifier] = function
        return outputs
