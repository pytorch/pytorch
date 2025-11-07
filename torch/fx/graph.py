# mypy: allow-untyped-defs
import builtins
import contextlib
import copy
import enum
import functools
import inspect
import keyword
import math
import os
import pprint
import re
import typing
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._C import _fx_map_arg as map_arg, _NodeIter
from torch.utils._dtype_abbrs import dtype_abbrs

from . import _pytree as fx_pytree
from ._compatibility import compatibility
from .immutable_collections import immutable_dict
from .node import _get_qualified_name, _type_repr, Argument, Node, Target


__all__ = ["PythonCode", "CodeGen", "Graph"]

if TYPE_CHECKING:
    from ._symbolic_trace import Tracer  # noqa: F401
    from .graph_module import GraphModule  # noqa: F401


# Mapping of builtins to their `typing` equivalent.
# (PEP585: See D68459095 test plan)
_origin_type_map = {
    list: typing.List,  # noqa: UP006
    dict: typing.Dict,  # noqa: UP006
    set: typing.Set,  # noqa: UP006
    frozenset: typing.FrozenSet,  # noqa: UP006
    tuple: typing.Tuple,  # noqa: UP006
}

_legal_ops = dict.fromkeys(
    ["call_function", "call_method", "get_attr", "call_module", "placeholder", "output"]
)


# Signature for functions thattransforms the body (`list[str]`) of the
# generated code
TransformCodeFunc = Callable[[list[str]], list[str]]


class _CustomBuiltin(NamedTuple):
    """Additional objs that we add to every graph's globals.

    The repr() for some standard library objects is not valid Python code without
    an import. For common objects of this sort, we bundle them in the globals of
    every FX graph.
    """

    # How to import this object from the standard library.
    import_str: str
    # The actual object, produced from that import string.
    obj: Any


# Combined dict of disallowed variable names so we can check with one lookup
_illegal_names = {k: object() for k in keyword.kwlist}
_illegal_names.update(builtins.__dict__)  # can't shadow a builtin name

_custom_builtins: dict[str, _CustomBuiltin] = {}


def _register_custom_builtin(name: str, import_str: str, obj: Any):
    _custom_builtins[name] = _CustomBuiltin(import_str, obj)
    _illegal_names[name] = obj


_register_custom_builtin("inf", "from math import inf", math.inf)
_register_custom_builtin("nan", "from math import nan", math.nan)
_register_custom_builtin("NoneType", "NoneType = type(None)", type(None))
_register_custom_builtin("torch", "import torch", torch)
_register_custom_builtin("device", "from torch import device", torch.device)
_register_custom_builtin("fx_pytree", "import torch.fx._pytree as fx_pytree", fx_pytree)
_register_custom_builtin("pytree", "import torch.utils._pytree as pytree", pytree)


def _is_magic(x: str) -> bool:
    return x.startswith("__") and x.endswith("__")


def _snake_case(s: str) -> str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    return _snake_case_sub(s).lower()


# Replace occurrences where a lowercase letter is followed by an uppercase letter
_snake_case_sub = functools.partial(re.compile(r"(?<=[a-z])([A-Z])").sub, r"_\1")

# Find chars that can't be in a Python identifier
_illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")

# Combined check for variable names:
# 1) Checks name is not empty
# 2) Checks first character is not a digit
# 3) Checks name has no illegal characters (_illegal_char_regex)
# 3) Splits off the number suffix (if present)
_name_regex = re.compile(r"^([a-zA-Z_][0-9a-zA-Z_]*?)(?:_(\d+))?$")

# starts with torch but does not start with torch._dynamo. or torch._inductor.
_torch_but_not_dynamo = re.compile(
    r"^torch(?:\.(?!_dynamo\.|_inductor\.)[^.]+)*$"
).fullmatch


def _is_from_torch(obj: Any) -> bool:
    module_name = getattr(obj, "__module__", None)
    if module_name is not None:
        return _torch_but_not_dynamo(module_name) is not None

    name = getattr(obj, "__name__", None)
    # exclude torch because torch.torch.torch.torch works. idk mang
    if name is not None and name != "torch":
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is obj:
                return True

    return False


class _Namespace:
    """A context for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """

    def __init__(self):
        self._obj_to_name: dict[Any, str] = {}
        self._used_names: set[str] = set()
        self._base_count: dict[str, int] = {}

    def create_name(self, candidate: str, obj: Optional[Any]) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        # optimistically check if candidate is already a valid name
        match = _name_regex.match(candidate)
        if match is None:
            # delete all characters that are illegal in a Python identifier
            candidate = _illegal_char_regex.sub("_", candidate)

            if not candidate:
                candidate = "_unnamed"

            if candidate[0].isdigit():
                candidate = f"_{candidate}"

            match = _name_regex.match(candidate)
            assert match is not None

        base, num = match.group(1, 2)
        if num is None or candidate in self._used_names:
            num = self._base_count.get(candidate, 0)
            if _illegal_names.get(candidate, obj) is not obj:
                num += 1
                candidate = f"{base}_{num}"
                # assume illegal names don't end in _\d so no need to check again
        else:
            num = int(num)

        while candidate in self._used_names:
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        if obj is not None:
            self._obj_to_name[obj] = candidate
        return candidate

    def associate_name_with_obj(self, name: str, obj: Any):
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already.
        """
        maybe_existing = self._obj_to_name.setdefault(obj, name)
        assert maybe_existing is name, "obj is already associated"

    def _rename_object(self, obj: Any, name: str):
        assert obj in self._obj_to_name
        self._obj_to_name[obj] = name
        self._used_names.add(name)


@compatibility(is_backward_compatible=True)
@dataclass
class PythonCode:
    """
    Represents all the information necessary to exec or save a graph as Python code.
    """

    # Python source code for the forward function definition.
    src: str
    # Values in global scope during execution of `src_def`.
    globals: dict[str, Any]
    # Optional mapping from the forward function's line number to
    # node index. Line number starts at the prologue (i.e. forward()).
    _lineno_map: Optional[dict[int, Optional[int]]]
    # The line number of prologue in fn_code
    _prologue_start: int = 0


def _format_target(base: str, target: str) -> str:
    elems = target.split(".")
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f"{r}.{e}"
    return r


class _InsertPoint:
    def __init__(self, graph, new_insert):
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        self.graph._insert = self.orig_insert


class _node_list:
    def __init__(self, graph: "Graph", direction: Literal["_prev", "_next"] = "_next"):
        assert direction in ("_next", "_prev")
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        return _NodeIter(self.graph._root, self.direction == "_prev")

    def __reversed__(self):
        return _node_list(self.graph, "_next" if self.direction == "_prev" else "_prev")


class _PyTreeInfo(NamedTuple):
    """
    Contains extra info stored when we're using Pytrees
    """

    orig_args: list[str]
    in_spec: pytree.TreeSpec
    out_spec: Optional[pytree.TreeSpec]


@dataclass(frozen=True)
class _ParsedStackTrace:
    """
    Represents the top-most frame of a parsed stack trace
    """

    file: str
    lineno: str
    name: str
    code: str

    def get_summary_str(self):
        return f"File: {self.file}:{self.lineno} in {self.name}, code: {self.code}"


# get File:lineno code from stack_trace
def _parse_stack_trace(stack_trace: str):
    if stack_trace is None:
        return None
    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    lines = stack_trace.strip().split("\n")
    # stacktrace should have innermost frame last, so we
    # iterate backwards to find the first line that starts
    # with 'File '
    for idx in range(len(lines) - 2, -1, -1):
        line = lines[idx].strip()
        matches = pattern.match(line)
        if matches:
            file = matches.group(1)
            lineno = matches.group(2)
            name = matches.group(3)
            # next line should be the code
            code = lines[idx + 1].strip()
            return _ParsedStackTrace(file, lineno, name, code)
    return None


@compatibility(is_backward_compatible=False)
class CodeGen:
    # This is an override hook so we can customize the SymNode printer.
    _sym_repr: Callable[["torch.types.PySymType"], str] = lambda x: repr(x)

    def __init__(self):
        self._body_transformer: Optional[TransformCodeFunc] = None
        self._func_name: str = "forward"

    def _format_multiline_args(self, args: list[str]) -> str:
        """Helper to format function arguments in expanded multiline format."""
        return "".join(self._format_single_arg(arg) for arg in args)

    def _format_single_arg(self, arg: str) -> str:
        """Helper to format a single argument with optional comment."""
        if "#" in arg:
            arg_part, comment_part = arg.split("#", 1)
            return f"    {arg_part.rstrip()},  # {comment_part.lstrip()}\n"
        else:
            return f"    {arg},\n"

    def _get_delimiters(self, container) -> tuple[str, str]:
        """Helper to get opening and closing delimiters for containers."""
        return ("(", ")") if isinstance(container, tuple) else ("[", "]")

    def _format_multiline_container(self, items, descs=None, prefix="") -> str:
        """Helper to format containers (lists/tuples) in multiline format."""
        ldelim, rdelim = self._get_delimiters(items)
        desc_trailers = self._get_desc_trailers(items, descs)

        return (
            f"{prefix}{ldelim}\n"
            + "".join(
                f"    {item},{trailer}\n" for item, trailer in zip(items, desc_trailers)
            )
            + f"{rdelim}"
        )

    def _get_desc_trailers(self, items, descs):
        """Helper to generate description trailers for items."""
        if descs is None:
            return [""] * len(items)
        return [f"  # {desc}" for desc in descs]

    def _call_method_with_signature_check(self, method, *args, **kwargs):
        """Helper to call a method with optional parameters based on signature."""
        sig = inspect.signature(method)
        # Filter kwargs to only include parameters that exist in the method signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return method(*args, **filtered_kwargs)

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        """
        Given the free variables and a return annotation, generates the beginning of the FX function.
        By default, `gen_fn_def(['a', 'b'], '') == 'def {self._func_name}(a, b):'`
        """
        # If the original function didn't have self as its first argument, we
        # would have added it.
        if len(free_vars) == 0 or free_vars[0] != "self":
            free_vars.insert(0, "self")

        if expanded_def:
            args_formatted = self._format_multiline_args(free_vars)
            return (
                f"def {self._func_name}(\n{args_formatted}){maybe_return_annotation}:"
            )
        else:
            return f"def {self._func_name}({', '.join(free_vars)}){maybe_return_annotation}:"

    def generate_output(
        self, output_args: Argument, *, descs: Optional[Any] = None
    ) -> str:
        """
        Given the output arguments, generates the return statement of the FX function.
        Note: The returned statement should not be indented.
        """
        if descs is not None and isinstance(output_args, (list, tuple)):
            return self._format_multiline_container(output_args, descs, "return ")
        else:
            return f"return {repr(output_args)}"

    def process_inputs(self, *args: Any) -> Any:
        """
        Transforms the inputs so that the graph can take them as arguments, as
        non-default codegen may result in the inputs to the function being
        different from the inputs to the graph.

        If the graph was directly runnable, this invariant should hold true
        `f.graph.process_outputs(f.graph(*f.graph.process_inputs(*inputs))) == f(*inputs)`
        """
        return args

    def process_outputs(self, outputs: Any) -> Any:
        """
        Transforms the outputs of the graph to be identical to the codegen.

        See ``process_inputs`` for more details.
        """
        return outputs

    def additional_globals(self) -> list[tuple[str, Any]]:
        """
        If your codegen uses extra global values, add tuples of (identifier,reference to the value) here.
        For example, return ['List', typing.List] if you need ``List`` in the global context.
        """
        return []

    def _gen_python_code(
        self,
        nodes,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        # Render each argument on its own line
        expanded_def: bool = False,
        record_func: bool = False,
    ) -> PythonCode:
        free_vars: list[str] = []
        body: list[str] = []
        globals_: dict[str, Any] = {}
        wrapped_fns: dict[str, None] = {}

        # Wrap string in list to pass by reference
        maybe_return_annotation: list[str] = [""]
        include_stride = include_stride or (
            os.environ.get("FX_GRAPH_SHOW_STRIDE", "0") == "1"
        )
        include_device = include_device or (
            os.environ.get("FX_GRAPH_SHOW_DEVICE", "0") == "1"
        )
        include_meta = os.environ.get("FX_GRAPH_SHOW_META", "0") == "1"

        def add_global(name_hint: str, obj: Any):
            """Add an obj to be tracked as a global.

            We call this for names that reference objects external to the
            Graph, like functions or types.

            Returns: the global name that should be used to reference 'obj' in generated source.
            """
            if (
                _is_from_torch(obj) and obj != torch.device
            ):  # to support registering torch.device
                # HACK: workaround for how torch custom ops are registered. We
                # can't import them like normal modules so they must retain their
                # fully qualified name.
                return _get_qualified_name(obj)

            # normalize the name hint to get a proper identifier
            global_name = namespace.create_name(name_hint, obj)

            if global_name in globals_:
                assert globals_[global_name] == obj
                return global_name
            globals_[global_name] = obj
            return global_name

        # Pre-fill the globals table with registered builtins.
        for name, (_, obj) in _custom_builtins.items():
            add_global(name, obj)

        def type_repr(o: Any):
            if o == ():
                # Empty tuple is used for empty tuple type annotation Tuple[()]
                return "()"

            typename = _type_repr(o)

            if origin_type := getattr(o, "__origin__", None):
                # list[...], typing.List[...], TensorType[...]

                if isinstance(o, typing._GenericAlias):  # type: ignore[attr-defined]
                    # This is a generic pre-PEP585 type, e.g. typing.List[torch.Tensor]
                    origin_type = _origin_type_map.get(origin_type, origin_type)

                origin_typename = add_global(_type_repr(origin_type), origin_type)

                if hasattr(o, "__args__") and o.__args__:
                    args = [type_repr(arg) for arg in o.__args__]
                    return f"{origin_typename}[{','.join(args)}]"
                else:
                    return origin_typename

            # Common case: this is a regular module name like 'foo.bar.baz'
            return add_global(typename, o)

        if colored:
            red = _color_fns["red"]
            dim_green = _color_fns["dim_green"]
            dim = _color_fns["dim"]
            dim_blue = _color_fns["dim_blue"]
            blue = _color_fns["blue"]
        else:
            red = _identity
            dim_green = _identity
            dim = _identity
            dim_blue = _identity
            blue = _identity

        def _get_repr(arg: Any) -> str:
            if isinstance(arg, Node):  # first because common
                return repr(arg)
            elif isinstance(arg, tuple) and hasattr(arg, "_fields"):
                # Handle NamedTuples (if it has `_fields`) via add_global.
                qualified_name = _get_qualified_name(type(arg))
                global_name = add_global(qualified_name, type(arg))
                return f"{global_name}{repr(tuple(arg))}"
            elif isinstance(
                arg, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
            ):
                qualified_name = _get_qualified_name(arg)
                global_name = add_global(qualified_name, arg)
                return f"{global_name}"
            elif isinstance(arg, enum.Enum):
                cls = arg.__class__
                clsname = add_global(cls.__name__, cls)
                return f"{clsname}.{arg.name}"
            elif isinstance(arg, torch.Tensor):
                size = list(arg.size())
                dtype = str(arg.dtype).split(".")[-1]
                return f"torch.Tensor(size={size}, dtype={dtype})"
            elif isinstance(arg, tuple):
                if len(arg) == 1:
                    return f"({_get_repr(arg[0])},)"
                else:
                    return "(" + ", ".join(_get_repr(a) for a in arg) + ")"
            elif isinstance(arg, list):
                return "[" + ", ".join(_get_repr(a) for a in arg) + "]"
            elif isinstance(arg, slice):
                return f"slice({_get_repr(arg.start)}, {_get_repr(arg.stop)}, {_get_repr(arg.step)})"
            else:
                return blue(repr(arg))

        def _format_args(
            args: tuple[Argument, ...], kwargs: dict[str, Argument]
        ) -> str:
            res = [_get_repr(a) for a in args]
            res.extend([f"{k} = {_get_repr(v)}" for k, v in kwargs.items()])
            return ", ".join(res)

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use: dict[Node, Node] = {}
        user_to_last_uses: dict[Node, list[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            for input_node in node._input_nodes:
                register_last_uses(input_node, node)

        def delete_unused_values(user: Node):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            if user.op == "placeholder":
                return
            if user.op == "output":
                body.append("\n")
                return
            nodes_to_delete = user_to_last_uses.get(user, [])

            if len(user.users.keys()) == 0:
                # This node is not used by any others. however it's also not
                # removed by DCE since side-effect. We want to free it's outputs
                # right after its execution done to save memory.
                nodes_to_delete.append(user)

            if len(nodes_to_delete):
                to_delete_str = " = ".join(
                    [repr(n) for n in nodes_to_delete] + ["None"]
                )
                body.append(f";  {dim(to_delete_str)}\n")
            else:
                body.append("\n")

        prev_summary_str = None

        def append_stacktrace_summary(node: Node):
            """
            Append a summary of the stacktrace to the generated code. This is
            useful for debugging.
            """
            nonlocal prev_summary_str

            if node.op not in {"placeholder", "output"}:
                annotation_str = ""
                annotation = node.meta.get("custom", {})
                if annotation:
                    annotation_str = f" Annotation: {annotation}"

                stack_trace_str = "No stacktrace found for following nodes"
                if stack_trace := node.stack_trace:
                    if parsed_stack_trace := _parse_stack_trace(stack_trace):
                        stack_trace_str = parsed_stack_trace.get_summary_str()

                summary_str = f"\n{dim(f'#{annotation_str} {stack_trace_str}')}\n"

                if summary_str != prev_summary_str:
                    prev_summary_str = summary_str
                    body.append(summary_str)

        def stringify_shape(shape: Iterable) -> str:
            return f"[{', '.join([str(x) for x in shape])}]"

        def emit_node(node: Node):
            maybe_type_annotation = (
                "" if node.type is None else f" : {type_repr(node.type)}"
            )
            maybe_comment = ""

            if verbose:
                # override annotation with more detailed information
                try:
                    from torch.distributed.tensor._api import DTensor, DTensorSpec

                    dtensorspec_format_shard_order_str = (
                        DTensorSpec.format_shard_order_str
                    )
                except ModuleNotFoundError:
                    DTensor = None  # type: ignore[assignment,misc]
                    dtensorspec_format_shard_order_str = None
                from torch.fx.experimental.proxy_tensor import py_sym_types
                from torch.fx.passes.shape_prop import TensorMetadata

                meta_val = node.meta.get(
                    "val",
                    node.meta.get("tensor_meta", node.meta.get("example_value", None)),
                )

                def _tensor_annotation(t: torch.Tensor) -> str:
                    stride = stringify_shape(t.stride()) if include_stride else ""
                    device = f"{t.device}" if include_device else ""
                    return (
                        f"{red(dtype_abbrs[t.dtype])}"
                        f"{blue(stringify_shape(t.shape))}"
                        f"{dim_blue(stride)}"
                        f"{dim_green(device)}"
                    )

                # use string as annotation, to make it valid python code
                if isinstance(meta_val, torch.Tensor) and meta_val.layout not in (
                    torch.sparse_csc,
                    torch.sparse_csr,
                ):
                    # Fake tensors cause tests to wobble, so do not custom print them.
                    is_plain = type(meta_val) is torch.Tensor or isinstance(
                        meta_val, torch._subclasses.FakeTensor
                    )
                    core = _tensor_annotation(meta_val)
                    if is_plain:
                        maybe_type_annotation = f': "{core}"'
                    elif type(meta_val) is DTensor:
                        assert dtensorspec_format_shard_order_str is not None
                        dtensor_meta = dtensorspec_format_shard_order_str(
                            meta_val._spec.placements,  # type: ignore[attr-defined]
                            meta_val._spec.shard_order,  # type: ignore[attr-defined]
                        )
                        cls = meta_val.__class__.__name__
                        maybe_type_annotation = (
                            f': "{cls}({core}, {dim_green(dtensor_meta)})"'
                        )
                    else:
                        cls = meta_val.__class__.__name__
                        maybe_type_annotation = f': "{cls}({core})"'

                elif isinstance(meta_val, py_sym_types):
                    val_str = CodeGen._sym_repr(meta_val)
                    maybe_type_annotation = f': "Sym({val_str})"'

                elif isinstance(meta_val, TensorMetadata):
                    maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'

            desc = None
            if expanded_def:
                desc = node.meta.get("desc", None)
                if desc is not None and node.op == "placeholder":
                    maybe_comment += f"  # {desc}"
                # output is handled specially

            if include_meta and hasattr(node, "meta") and node.meta:
                body.append('"""\n')
                for k, v in node.meta.items():
                    # use str over repr since repr is susceptible to sympy
                    # errors such as "cannot determine truth value of Relational"
                    # Pretty print the high-level dict with str() for values
                    body.append(
                        f"{k}: {pprint.pformat(str(v), width=80, compact=True)}\n"
                    )
                body.append('"""\n')

            if node.op == "placeholder":
                assert isinstance(node.target, str)
                maybe_default_arg = (
                    "" if not node.args else f" = {_get_repr(node.args[0])}"
                )
                free_vars.append(
                    f"{node.target}{maybe_type_annotation}{maybe_default_arg}{maybe_comment}"
                )
                raw_name = node.target.replace("*", "")
                if raw_name != repr(node):
                    body.append(f"{repr(node)} = {raw_name}\n")
                return
            elif node.op == "call_method":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.target)}"
                    f"({_format_args(node.args[1:], node.kwargs)})"
                )
                return
            elif node.op == "call_function":
                assert callable(node.target)
                # pretty print operators
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in magic_methods
                ):
                    assert isinstance(node.args, tuple)
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = "
                        f"{magic_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))}"
                    )
                    return

                # pretty print inplace operators; required for jit.script to work properly
                # not currently supported in normal FX graphs, but generated by torchdynamo
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in inplace_methods
                ):
                    body.append(
                        f"{inplace_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))};  "
                        f"{repr(node)}{maybe_type_annotation} = {_get_repr(node.args[0])}"
                    )
                    return

                qualified_name = _get_qualified_name(node.target)
                global_name = add_global(qualified_name, node.target)
                # special case for getattr: node.args could be 2-argument or 3-argument
                # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                if (
                    global_name == "getattr"
                    and isinstance(node.args, tuple)
                    and isinstance(node.args[1], str)
                    and node.args[1].isidentifier()
                    and len(node.args) == 2
                ):
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.args[1])}"
                    )
                    return
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})"
                )
                if node.meta.get("is_wrapped", False):
                    wrapped_fns.setdefault(global_name)
                return
            elif node.op == "call_module":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = "
                    f"{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})"
                )
                return
            elif node.op == "get_attr":
                assert isinstance(node.target, str)
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}"
                )
                return
            elif node.op == "output":
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                body.append(
                    self._call_method_with_signature_check(
                        self.generate_output,
                        node.args[0],
                        descs=desc if expanded_def else None,
                    )
                )
                return
            raise NotImplementedError(f"node: {node.op} {node.target}")

        if record_func:
            body.append(
                "_rf = torch._C._profiler._RecordFunctionFast('## ENTER_GRAPH_PLACEHOLDER_KEY ##'); _rf.__enter__()\n"
            )
        for i, node in enumerate(nodes):
            # NOTE: emit_node does not emit a string with newline. It depends
            # on delete_unused_values to append one
            if verbose:
                append_stacktrace_summary(node)
            # emit a counter comment to keep track of
            # node index, which will be deleted later
            # after going through _body_transformer
            body.append(f"# COUNTER: {i}\n")
            do_record = record_func and node.op in (
                "call_function",
                "call_method",
                "call_module",
            )
            if do_record:
                # The double hash ## convention is used by post-processing to find the fx markers
                body.append(
                    f"_rf_{node.name} = torch._C._profiler._RecordFunctionFast('## {i} ##'); _rf_{node.name}.__enter__()\n"
                )
            emit_node(node)
            delete_unused_values(node)
            if do_record:
                body.append(f"_rf_{node.name}.__exit__(None, None, None)\n")
        if record_func:
            body.append("_rf.__exit__(None, None, None)\n")

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append("pass\n")

        if len(wrapped_fns) > 0:
            wrap_name = add_global("wrap", torch.fx.wrap)
            wrap_stmts = "\n".join([f'{wrap_name}("{name}")' for name in wrapped_fns])
        else:
            wrap_stmts = ""

        if self._body_transformer:
            body = self._body_transformer(body)

        for name, value in self.additional_globals():
            add_global(name, value)

        prologue = self._call_method_with_signature_check(
            self.gen_fn_def,
            free_vars,
            maybe_return_annotation[0],
            expanded_def=expanded_def,
        )

        # remove counter and generate lineno to node index mapping
        lineno_map: dict[int, Optional[int]] = {}
        prologue_len = prologue.count("\n") + 1
        new_lines: list[str] = []
        cur_idx = None
        for line in "".join(body).split("\n"):
            counter = _counter_regexp.search(line)
            if counter is not None:
                cur_idx = int(counter.group(1))
            else:
                lineno_map[len(new_lines) + prologue_len] = cur_idx
                new_lines.append(line)

        code = "\n".join(new_lines).lstrip("\n")
        code = "\n".join("    " + line for line in code.split("\n"))

        fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
        # The +4 accounts for the empty lines before prologue in fn_code
        prologue_start = wrap_stmts.count("\n") + 4
        return PythonCode(
            fn_code,
            globals_,
            _lineno_map=lineno_map,
            _prologue_start=prologue_start,
        )


# Ideally, we'd like to refactor all of the pytree logic into this codegen
# class. Unfortunately, there are 3 areas we currently need extra logic in FX.
# 1. In the initial symbolic trace, the pytree logic is tied up with `concrete_args`.
# 2. In the FX graph, we need to access 2 attributes - in_spec and out_spec.
#    Since we can't access .graph within the FX forward, we need to copy the attribute to the module.
# 3. We currently can't register the pytree imports with `add_global` - not sure why.
class _BoxedCodeGen(CodeGen):
    """
    CodeGen subclass that generates code using the "boxed" calling convention.

    The boxed calling convention takes a single list argument and clears it
    after extracting the arguments, which allows for early deallocation of
    input tensors.
    """

    def gen_fn_def(
        self, free_vars, maybe_return_annotation, *, expanded_def: bool = False
    ):
        """
        Generate function definition for boxed calling convention.

        Instead of taking individual arguments, the generated function takes
        a single 'args_list' parameter, extracts placeholder values from it,
        and clears the list.
        """
        # Generate the function signature with args_list parameter
        fn_def = f"def {self._func_name}(self, args_list){maybe_return_annotation}:"

        if free_vars:
            # This is horribly manual but we don't get the "raw" free vars
            # without a bigger refactor.
            placeholder_vars = [
                v.split(":")[0].split("=")[0].strip() for v in free_vars if v != "self"
            ]

            if placeholder_vars:
                fn_def += "\n    args_iter = iter(args_list)"
                for var in placeholder_vars:
                    fn_def += f"\n    {var} = next(args_iter)"
                fn_def += "\n    args_list.clear()"

        return fn_def


class _PyTreeCodeGen(CodeGen):
    def __init__(self, pytree_info: _PyTreeInfo):
        super().__init__()
        self.pytree_info: _PyTreeInfo = pytree_info

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = pytree.arg_tree_leaves(*inputs)
        return flat_args

    def process_outputs(self, out: Any) -> Any:
        if self.pytree_info is None or self.pytree_info.out_spec is None:
            return out
        if not isinstance(out, (list, tuple)):
            out = [out]
        assert self.pytree_info.out_spec is not None
        return pytree.tree_unflatten(out, self.pytree_info.out_spec)

    def _format_annotations(self, free_vars: list[str], expanded_def: bool) -> str:
        """Helper to format annotations for variables in pytree codegen."""
        if not free_vars:
            return ""

        has_annotation = [x for x in free_vars if ":" in x]
        if not has_annotation:
            return ""

        if expanded_def:
            return "\n    " + "\n    ".join(has_annotation)
        else:
            return "\n    " + "".join(x + "; " for x in has_annotation) + "\n"

    def gen_var_bindings(self, fn_args, free_vars, expanded_def) -> str:
        in_spec = self.pytree_info.in_spec
        # when kwargs is present, in_spec is tuple(args, kwargs)
        has_args_kwargs_tuple = (
            in_spec.type is tuple
            and in_spec.num_children == 2
            and in_spec.child(0).type is tuple
            and in_spec.child(1).type is dict
        )
        fn_kwargs = "{}"
        fn_signature = f"[{', '.join(fn_args)}], self._in_spec"
        if has_args_kwargs_tuple:
            count_args = in_spec.child(0).num_children
            fn_args = self.pytree_info.orig_args[:count_args]
            fn_kwargs = (
                "{"
                + ", ".join(
                    f"'{k}':{v}"
                    for k, v in zip(
                        in_spec.child(1).context,
                        self.pytree_info.orig_args[count_args:],
                    )
                )
                + "}"
            )
            fn_signature = f"([{', '.join(fn_args)}], {fn_kwargs}), self._in_spec"

        # in Python, `var1: annotation1, var2: annotation2 = function_call()` is invalid.
        # we need to split it to two lines:
        # one for annotation: `var1: annotation1; var2: annotation2;` (note the semicolon)
        # one for code: `var1, var2, = function_call()`
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        bindings = self._format_annotations(free_vars, expanded_def)
        bindings += f"""
    {", ".join(without_annotation)}, = fx_pytree.tree_flatten_spec({fn_signature})"""
        return bindings

    def gen_fn_def(
        self, free_vars, maybe_return_annotation, *, expanded_def: bool = False
    ):
        # Given a user function/model:
        #   myargs = [myargs0, myargs1]
        #   mykwargs = {'mykwargs0': ..., 'mykwargs1': ...}
        #   def forward(self, mypos, *myargs, mykey=None, **mykwargs):
        #
        # The generated code flattens all keywords into positional arguments for `forward()`
        #   e.g forward(self, mypos, myargs0, myargs1, mykey, mykwargs0, mykwargs1):
        #
        # Within `forward`, `tree_flatten_spec``still parses args and kwargs separately
        #   e.g. tree_flatten_spec(([mypos, myargs0, myargs1],
        #                           {'mykey':mykey, 'mykwargs0':mykwargs0, 'mykwargs1':mykwargs1}),
        #                          self._in_spec)
        #
        # If the user function/model does not have keywords, the dict is suppressed from tree_flatten_spec
        #   e.g. tree_flatten_spec([mypos, myargs0, myargs1]), self._in_spec)
        if self.pytree_info is None:
            return super().gen_fn_def(
                free_vars, maybe_return_annotation, expanded_def=expanded_def
            )

        fn_args = self.pytree_info.orig_args
        has_orig_self = (fn_args[0] == "self") if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, "self")
        fn_definition = super().gen_fn_def(
            fn_args[:], maybe_return_annotation, expanded_def=expanded_def
        )

        if len(free_vars) > 0:  # pytree has placeholders in it
            fn_definition += self.gen_var_bindings(fn_args, free_vars, expanded_def)
        return fn_definition

    def generate_output(self, output_args, *, descs: Optional[Any] = None):
        if self.pytree_info and self.pytree_info.out_spec:
            if descs is not None and isinstance(output_args, (list, tuple)):
                return (
                    self._format_multiline_container(
                        output_args, descs, "return pytree.tree_unflatten("
                    )
                    + ", self._out_spec)"
                )
            else:
                return (
                    f"return pytree.tree_unflatten({repr(output_args)}, self._out_spec)"
                )
        else:
            return super().generate_output(output_args, descs=descs)


class _ExportCodeGen(_PyTreeCodeGen):
    def __init__(
        self,
        pytree_info: _PyTreeInfo,
        in_shuffle_graph: "GraphModule",
        out_shuffle_graph: "GraphModule",
        tree_leaf_names: list[str],
        root: Optional[torch.nn.Module],
    ):
        super().__init__(pytree_info)
        self.in_shuffle_graph = in_shuffle_graph
        self.out_shuffle_graph = out_shuffle_graph
        self.tree_leaf_names = tree_leaf_names
        self.root = root

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = super().process_inputs(*inputs)
        if self.root is not None:
            flat_args = (self.root, *flat_args)
        self.flat_args = flat_args
        return self.in_shuffle_graph(*flat_args)

    def process_outputs(self, out: Any) -> Any:
        flat_outs = self.out_shuffle_graph(*self.flat_args, *out)
        del self.flat_args
        ret = super().process_outputs(flat_outs)
        return ret

    def gen_fn_def(self, *args, **kwargs) -> str:
        fn_def = super().gen_fn_def(*args, **kwargs)
        return fn_def

    def gen_var_bindings(self, fn_args, free_vars, expanded_def) -> str:
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        fn_signature: str = f"{', '.join(fn_args)}"
        if self.root is not None:
            fn_signature = f"self, {fn_signature}"
        return f"""
    {", ".join(self.tree_leaf_names)}, = pytree.tree_leaves(({fn_signature},))
    {", ".join(without_annotation)}, = self._in_shuffle_graph({", ".join(self.tree_leaf_names)})"""

    def generate_output(self, output_args, *args, **kwargs) -> str:
        output = f"self._out_shuffle_graph({', '.join(self.tree_leaf_names)}, {', '.join([str(a) for a in output_args])})"
        return f"return pytree.tree_unflatten({output}, self._out_spec)"


class _FindNodesLookupTable:
    """
    Side table for the graph for the purpose of doing fast queries
    """

    def __init__(self):
        self.table: dict[tuple[str, Optional[Target]], dict[Node, None]] = defaultdict(
            dict
        )

    def _key(self, node) -> tuple[str, Optional[Target]]:
        return (node.op, node.target if node.op == "call_function" else None)

    def __contains__(self, node) -> bool:
        return node in self.table[self._key(node)]

    def insert(self, node: Node) -> None:
        self.table[self._key(node)][node] = None

    def remove(self, node: Node) -> None:
        self.table[self._key(node)].pop(node)

    def find_nodes(self, *, op: str, target: Optional["Target"] = None):
        if op == "call_function":
            assert target is not None
            return [*self.table[(op, target)].keys()]

        if target is None:
            return [*self.table[(op, None)].keys()]

        # op is call_method, get_attr, call_module
        return [node for node in self.table[(op, None)].keys() if node.target == target]


@compatibility(is_backward_compatible=True)
class Graph:
    """
    ``Graph`` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of ``Node`` s, each representing callsites (or other
    syntactic constructs). The list of ``Node`` s, taken together, constitute a
    valid Python function.

    For example, the following code

    .. code-block:: python

        import torch
        import torch.fx


        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return torch.topk(
                    torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
                )


        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

    Will produce the following Graph::

        print(gm.graph)

    .. code-block:: text

        graph(x):
            %linear_weight : [num_users=1] = self.linear.weight
            %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
            %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
            %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
            %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
            %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
            return topk_1

    For the semantics of operations represented in the ``Graph``, please see :class:`Node`.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        owning_module: Optional["GraphModule"] = None,
        tracer_cls: Optional[type["Tracer"]] = None,
        tracer_extras: Optional[dict[str, Any]] = None,
    ):
        """
        Construct an empty Graph.
        """
        self._root: Node = Node(self, "", "root", "", (), {})
        self._used_names: dict[str, int] = {}  # base name -> number
        self._insert = self._root.prepend
        self._len = 0
        self._graph_namespace = _Namespace()
        self._owning_module = owning_module
        self._tracer_cls = tracer_cls
        self._tracer_extras = tracer_extras
        self._codegen = CodeGen()
        self._co_fields: dict[str, Any] = {}
        self._find_nodes_lookup_table = _FindNodesLookupTable()

    @property
    def owning_module(self):
        return self._owning_module

    @owning_module.setter
    def owning_module(self, mod: Optional["GraphModule"]):
        self._owning_module = mod

    @property
    def nodes(self) -> _node_list:
        """
        Get the list of Nodes that constitute this Graph.

        Note that this ``Node`` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.

        Returns:

            A doubly-linked list of Nodes. Note that ``reversed`` can be called on
            this list to switch iteration order.
        """
        return _node_list(self)

    @compatibility(is_backward_compatible=False)
    def output_node(self) -> Node:
        output_node = next(iter(reversed(self.nodes)))
        assert output_node.op == "output"
        return output_node

    @compatibility(is_backward_compatible=False)
    def find_nodes(
        self, *, op: str, target: Optional["Target"] = None, sort: bool = True
    ):
        """
        Allows for fast query of nodes

        Args:

            op (str): the name of the operation

            target (Optional[Target]): the target of the node. For call_function,
                the target is required. For other ops, the target is optional.

            sort (bool): whether to return nodes in the order they appear on
                         on the graph.

        Returns:

            Iterable of nodes with the requested op and target.
        """
        node_list = self._find_nodes_lookup_table.find_nodes(op=op, target=target)
        if sort:
            return sorted(node_list)
        return node_list

    @compatibility(is_backward_compatible=True)
    def graph_copy(
        self, g: "Graph", val_map: dict[Node, Node], return_output_node=False
    ) -> "Optional[Argument]":
        """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
        for node in g.nodes:
            if node in val_map:
                continue
            if node.op == "output":
                rv = map_arg(node.args[0], lambda n: val_map[n])
                return rv if not return_output_node else (rv, node)
            val_map[node] = self.node_copy(node, lambda n: val_map[n])
        return None

    def __deepcopy__(self, memo=None) -> "Graph":
        """
        Explicitly implement __deepcopy__ to prevent excessive recursion depth
        from the default implementation. This uses graph_copy to copy the nodes
        in an iterative way, rather than recursive. It also populates the
        memoization table to prevent unnecessary copies (e.g. references to
        nodes or other parts of the Graph from a custom GraphModule implementation.
        """
        memo = memo if memo else {}
        g = Graph(tracer_cls=self._tracer_cls)
        output_vals = g.graph_copy(self, val_map=memo, return_output_node=True)
        g._codegen = copy.deepcopy(self._codegen)
        if output_vals is not None:
            assert isinstance(output_vals, tuple)
            output_val, old_output_node = output_vals
            new_output_node = g.output(
                output_val, type_expr=getattr(old_output_node, "type", None)
            )
            new_output_node.meta = copy.copy(old_output_node.meta)
        return g

    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        op: str,
        target: "Target",
        args: Optional[tuple["Argument", ...]] = None,
        kwargs: Optional[dict[str, "Argument"]] = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Create a ``Node`` and add it to the ``Graph`` at the current insert-point.
        Note that the current insert-point can be set via :meth:`Graph.inserting_before`
        and :meth:`Graph.inserting_after`.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.

            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.

            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node

            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted node.
        """
        # `target in _legal_ops` is checked in Node.__init__
        if not args:
            args = ()
        else:
            assert isinstance(args, tuple), "args must be a tuple"
        if not kwargs:
            kwargs = immutable_dict()
        else:
            assert isinstance(kwargs, dict), "kwargs must be a dict"

        candidate = name if name is not None else self._target_to_str(target)
        name = self._graph_namespace.create_name(candidate, None)
        n = Node(self, name, op, target, args, kwargs, type_expr)

        if (
            self.owning_module is not None
            and getattr(self.owning_module, "_create_node_hooks", None) is not None
        ):
            for f in self.owning_module._create_node_hooks:
                f(n)

        self._graph_namespace.associate_name_with_obj(name, n)

        self._insert(n)
        self._find_nodes_lookup_table.insert(n)
        self._len += 1
        return n

    @compatibility(is_backward_compatible=False)
    def process_inputs(self, *args):
        """
        Processes args so that they can be passed to the FX graph.
        """
        return self._codegen.process_inputs(*args)

    @compatibility(is_backward_compatible=False)
    def process_outputs(self, out):
        return self._codegen.process_outputs(out)

    @compatibility(is_backward_compatible=True)
    def erase_node(self, to_erase: Node) -> None:
        """
        Erases a ``Node`` from the ``Graph``. Throws an exception if
        there are still users of that node in the ``Graph``.

        Args:

            to_erase (Node): The ``Node`` to erase from the ``Graph``.
        """
        if len(to_erase.users) > 0:
            raise RuntimeError(
                f"Tried to erase Node {to_erase} but it still had {len(to_erase.users)} "
                f"users in the graph: {to_erase.users}!"
            )
        if to_erase.graph != self:
            raise RuntimeError(f"Attempting to remove {to_erase} from wrong graph!")
        if to_erase._erased:
            warnings.warn(f"erase_node({to_erase}) on an already erased node")
            return

        if (
            self.owning_module is not None
            and getattr(self.owning_module, "_erase_node_hooks", None) is not None
        ):
            for f in self.owning_module._erase_node_hooks:
                f(to_erase)

        self._find_nodes_lookup_table.remove(to_erase)
        # pyrefly: ignore [missing-attribute]
        to_erase._remove_from_list()
        to_erase._erased = True  # iterators may retain handles to erased nodes
        self._len -= 1

        # Null out this Node's argument nodes so that the Nodes referred to
        # can update their ``users`` accordingly
        to_erase._update_args_kwargs(
            map_arg(to_erase._args, lambda n: None),
            map_arg(to_erase._kwargs, lambda n: None),
        )

    @compatibility(is_backward_compatible=True)
    def inserting_before(self, n: Optional[Node] = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_before(n):
                ...  # inserting before node n
            ...  # insert point restored to what it was previously
            g.inserting_before(n)  #  set the insert point permanently

        Args:

            n (Optional[Node]): The node before which to insert. If None this will insert before
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_after(self._root)
        assert n.graph == self, "Node to insert before is not in graph."
        return _InsertPoint(self, n.prepend)

    @compatibility(is_backward_compatible=True)
    def inserting_after(self, n: Optional[Node] = None):
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_after(n):
                ...  # inserting after node n
            ...  # insert point restored to what it was previously
            g.inserting_after(n)  #  set the insert point permanently

        Args:

            n (Optional[Node]): The node before which to insert. If None this will insert after
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_before(self._root)
        assert n.graph == self, "Node to insert after is not in graph."
        return _InsertPoint(self, n.append)

    @compatibility(is_backward_compatible=True)
    def placeholder(
        self,
        name: str,
        type_expr: Optional[Any] = None,
        default_value: Any = inspect.Signature.empty,
    ) -> Node:
        """
        Insert a ``placeholder`` node into the Graph. A ``placeholder`` represents
        a function input.

        Args:

            name (str): A name for the input value. This corresponds to the name
                of the positional argument to the function this ``Graph`` represents.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have. This is needed in some
                cases for proper code generation (e.g. when the function is used
                subsequently in TorchScript compilation).

            default_value (Any): The default value this function argument should take
                on. NOTE: to allow for `None` as a default value, `inspect.Signature.empty`
                should be passed as this argument to specify that the parameter does _not_
                have a default value.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        args = () if default_value is inspect.Signature.empty else (default_value,)
        return self.create_node("placeholder", name, args=args, type_expr=type_expr)

    @compatibility(is_backward_compatible=True)
    def get_attr(self, qualified_name: str, type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``get_attr`` node into the Graph. A ``get_attr`` ``Node`` represents the
        fetch of an attribute from the ``Module`` hierarchy.

        Args:

            qualified_name (str): the fully-qualified name of the attribute to be retrieved.
                For example, if the traced Module has a submodule named ``foo``, which has a
                submodule named ``bar``, which has an attribute named ``baz``, the qualified
                name ``foo.bar.baz`` should be passed as ``qualified_name``.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.


        Returns:

            The newly-created and inserted ``get_attr`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """

        def _get_attr_reference_exists(
            mod: torch.nn.Module, qualified_name: str
        ) -> bool:
            module_path, _, name = qualified_name.rpartition(".")

            try:
                submod: torch.nn.Module = mod.get_submodule(module_path)
            except AttributeError:
                warnings.warn(f"Failed to fetch module {module_path}!")
                return False

            if not hasattr(submod, name):
                return False

            res = getattr(submod, name)

            if (
                not isinstance(res, torch.nn.Module)
                and not isinstance(res, torch.nn.Parameter)
                and name not in submod._buffers
            ):
                return False

            return True

        if self.owning_module and not _get_attr_reference_exists(
            self.owning_module, qualified_name
        ):
            warnings.warn(
                "Attempted to insert a get_attr Node with no "
                "underlying reference in the owning "
                "GraphModule! Call "
                "GraphModule.add_submodule to add the "
                "necessary submodule, "
                "GraphModule.add_parameter to add the "
                "necessary Parameter, or "
                "nn.Module.register_buffer to add the "
                "necessary buffer",
                stacklevel=2,
            )
        return self.create_node("get_attr", qualified_name, type_expr=type_expr)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self,
        module_name: str,
        args: Optional[tuple["Argument", ...]] = None,
        kwargs: Optional[dict[str, "Argument"]] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Insert a ``call_module`` ``Node`` into the ``Graph``. A ``call_module`` node
        represents a call to the forward() function of a ``Module`` in the ``Module``
        hierarchy.

        Args:

            module_name (str): The qualified name of the ``Module`` in the ``Module``
                hierarchy to be called. For example, if the traced ``Module`` has a
                submodule named ``foo``, which has a submodule named ``bar``, the
                qualified name ``foo.bar`` should be passed as ``module_name`` to
                call that module.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this should *not* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted ``call_module`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        if self.owning_module and self.owning_module.get_submodule(module_name) is None:
            warnings.warn(
                "Attempted to insert a call_module Node with "
                "no underlying reference in the owning "
                "GraphModule! Call "
                "GraphModule.add_submodule to add the "
                "necessary submodule"
            )
        return self.create_node(
            "call_module", module_name, args, kwargs, type_expr=type_expr
        )

    @compatibility(is_backward_compatible=True)
    def call_method(
        self,
        method_name: str,
        args: Optional[tuple["Argument", ...]] = None,
        kwargs: Optional[dict[str, "Argument"]] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Insert a ``call_method`` ``Node`` into the ``Graph``. A ``call_method`` node
        represents a call to a given method on the 0th element of ``args``.

        Args:

            method_name (str): The name of the method to apply to the self argument.
                For example, if args[0] is a ``Node`` representing a ``Tensor``,
                then to call ``relu()`` on that ``Tensor``, pass ``relu`` to ``method_name``.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this *should* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly created and inserted ``call_method`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node(
            "call_method", method_name, args, kwargs, type_expr=type_expr
        )

    @compatibility(is_backward_compatible=True)
    def call_function(
        self,
        the_function: Callable[..., Any],
        args: Optional[tuple["Argument", ...]] = None,
        kwargs: Optional[dict[str, "Argument"]] = None,
        type_expr: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> Node:
        """
        Insert a ``call_function`` ``Node`` into the ``Graph``. A ``call_function`` node
        represents a call to a Python callable, specified by ``the_function``.

        Args:

            the_function (Callable[..., Any]): The function to be called. Can be any PyTorch
                operator, Python function, or member of the ``builtins`` or ``operator``
                namespaces.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called function.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called function

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

            name (Optional[str]): The name of the node. If not specified, set to None

        Returns:

            The newly created and inserted ``call_function`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node(
            "call_function", the_function, args, kwargs, name=name, type_expr=type_expr
        )

    @compatibility(is_backward_compatible=True)
    def node_copy(
        self, node: Node, arg_transform: Callable[[Node], "Argument"] = lambda x: x
    ) -> Node:
        """
        Copy a node from one graph into another. ``arg_transform`` needs to transform arguments from
        the graph of node to the graph of self. Example::

            # Copying all the nodes in `g` into `new_graph`
            g: torch.fx.Graph = ...
            new_graph = torch.fx.graph()
            value_remap = {}
            for node in g.nodes:
                value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

        Args:

            node (Node): The node to copy into ``self``.

            arg_transform (Callable[[Node], Argument]): A function that transforms
                ``Node`` arguments in node's ``args`` and ``kwargs`` into the
                equivalent argument in ``self``. In the simplest case, this should
                retrieve a value out of a table mapping Nodes in the original
                graph to ``self``.
        """
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        result_node = self.create_node(
            node.op, node.target, args, kwargs, node.name, node.type
        )
        result_node.meta = copy.copy(node.meta)
        return result_node

    @compatibility(is_backward_compatible=True)
    def output(self, result: "Argument", type_expr: Optional[Any] = None):
        """
        Insert an ``output`` ``Node`` into the ``Graph``. An ``output`` node represents
        a ``return`` statement in Python code. ``result`` is the value that should
        be returned.

        Args:

            result (Argument): The value to be returned.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        .. note::

            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        return self.create_node(
            op="output", target="output", args=(result,), type_expr=type_expr
        )

    def _target_to_str(self, target: Optional[Target]) -> str:
        if callable(target):
            op = target.__name__
        else:
            assert isinstance(target, str)
            op = target
            if _is_magic(op):
                op = op[2:-2]
        op = _snake_case(op)
        return op

    @compatibility(is_backward_compatible=True)
    def python_code(
        self,
        root_module: str,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
    ) -> PythonCode:
        """
        Turn this ``Graph`` into valid Python code.

        Args:

            root_module (str): The name of the root module on which to look-up
                qualified name targets. This is usually 'self'.

        Returns:

            A PythonCode object, consisting of two fields:
                src: the Python source code representing the object
                globals: a dictionary of global names in `src` -> the objects that they reference.
        """
        # NOTE: [Graph Namespaces]
        #
        # There are two types of symbols in generated Python source code:
        # locals and globals.
        #   Locals are locally defined by the output of a node in the Graph.
        #   Globals are references to external objects, like functions or types.
        #
        # When generating Python code, we need to make sure to name things
        # appropriately. In particular:
        # - All names should be unique, to avoid weird shadowing bugs.
        # - These names need to be consistent, e.g. a object should always be
        #   referenced by the same name.
        #
        # To do this, we create a new namespace just for this source. All names
        # that get printed must come from this namespace.
        #
        # Why can't we reuse node.name? Because it was generated within the
        # namespace `self._graph_namespace`. In order to provide uniqueness
        # over both locals (node.name) *and* globals, we create a completely
        # new namespace to put all identifiers in.
        namespace = _Namespace()

        # Override Node's repr to generate a valid name within our namespace.
        # Since repr() is designed to produce a valid Python expression, it
        # makes sense to reuse it. This way, it's easy to print something like
        # Tuple[Node, Node] by simply calling repr() on it. Node's __repr__ is
        # implemented cooperatively to allow this.
        def node_repr(n: Node):
            return namespace.create_name(n.name, n)

        @contextmanager
        def override_node_repr(graph: Graph):
            orig_repr_fns = {}
            for node in graph.nodes:
                orig_repr_fns[node] = node._repr_fn
                node._repr_fn = node_repr
            try:
                yield None
            finally:
                # restore the original repr functions
                for node in graph.nodes:
                    node._repr_fn = orig_repr_fns[node]

        with override_node_repr(self):
            return self._python_code(
                root_module,
                namespace,
                verbose=verbose,
                include_stride=include_stride,
                include_device=include_device,
                colored=colored,
                expanded_def=expanded_def,
                record_func=record_func,
            )

    def _python_code(
        self,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
    ) -> PythonCode:
        return self._codegen._gen_python_code(
            self.nodes,
            root_module,
            namespace,
            verbose=verbose,
            include_stride=include_stride,
            include_device=include_device,
            colored=colored,
            expanded_def=expanded_def,
            record_func=record_func,
        )

    def __str__(self) -> str:
        """
        Return a human-readable (not machine-readable) string representation
        of this Graph
        """
        placeholder_names: list[str] = []
        # This is a one-element array just so ``format_node`` can modify the closed
        # over value
        maybe_return_typename: list[str] = [""]

        node_strs = [node.format_node(placeholder_names) for node in self.nodes]
        param_str = ", ".join(placeholder_names)
        s = f"graph({param_str}){maybe_return_typename[0]}:"
        for node_str in node_strs:
            if node_str:
                s += "\n    " + node_str
        return s

    @compatibility(is_backward_compatible=True)
    def print_tabular(self):
        """
        Prints the intermediate representation of the graph in tabular
        format. Note that this API requires the ``tabulate`` module to be
        installed.
        """
        try:
            from tabulate import tabulate
        except ImportError:
            print(
                "`print_tabular` relies on the library `tabulate`, "
                "which could not be found on this machine. Run `pip "
                "install tabulate` to install the library."
            )
            raise

        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in self.nodes]
        print(
            tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"])
        )

    @compatibility(is_backward_compatible=True)
    def lint(self):
        """
        Runs various checks on this Graph to make sure it is well-formed. In
        particular:
        - Checks Nodes have correct ownership (owned by this graph)
        - Checks Nodes appear in topological order
        - If this Graph has an owning GraphModule, checks that targets
        exist in that GraphModule
        """

        # Check topo order
        def check_arg(arg: Node, n: Optional[Node] = None) -> None:
            context_str = f" of Node '{n}' " if n else " "
            if arg.graph is not self:
                raise RuntimeError(
                    f"Argument '{arg}'{context_str}does not belong to this Graph, "
                    f"but was used as an argument! If you are copying nodes from another graph, make "
                    f"sure to use ``arg_transform`` on node_copy() to remap values\n{self}"
                )
            if arg not in seen_values:
                raise RuntimeError(
                    f"Argument '{arg}'{context_str}was used before it has been "
                    f"defined! Please check that Nodes in the graph are topologically ordered\n{self}"
                )

        seen_names: set[str] = set()
        seen_values: set[Node] = set()
        for node in self.nodes:
            if node.op not in _legal_ops:
                raise RuntimeError(f"Node {node} had unknown opcode {node.op}!")
            if node.graph is not self:
                raise RuntimeError(f"Node '{node}' does not belong to this Graph!")
            if node not in self._find_nodes_lookup_table:
                raise RuntimeError(f"Node '{node}' is not added to the side table")
            for arg in node._input_nodes:
                check_arg(arg, node)
            seen_values.add(node)

            if node.name in seen_names:
                raise RuntimeError(f"Node redefined name {node.name}!")
            seen_names.add(node.name)

        # Check targets are legit
        if self.owning_module:
            for node in self.nodes:
                if node.op == "call_function":
                    if not callable(node.target):
                        raise ValueError(
                            f"Node {node} target {node.target} has type {torch.typename(node.target)} but "
                            "a Callable is expected"
                        )
                else:
                    if not isinstance(node.target, str):
                        raise ValueError(
                            f"Node {node} target {node.target} has type {torch.typename(node.target)} but "
                            "a str is expected"
                        )
                if node.op in ["get_attr", "call_module"]:
                    # pyrefly: ignore [missing-attribute]
                    target_atoms = node.target.split(".")
                    m_itr = self.owning_module
                    for i, atom in enumerate(target_atoms):
                        new_m_itr = getattr(m_itr, atom, None)
                        seen_qualname = ".".join(target_atoms[:i])
                        if new_m_itr is None:
                            raise RuntimeError(
                                f"Node {node} target {node.target} references nonexistent attribute "
                                f"{atom} of {seen_qualname}"
                            )
                        if node.op == "call_module" and not isinstance(
                            new_m_itr, torch.nn.Module
                        ):
                            raise RuntimeError(
                                f"Node {node} target {node.target} {atom} of {seen_qualname} does "
                                "not reference an nn.Module"
                            )

                        m_itr = new_m_itr

    @compatibility(is_backward_compatible=True)
    def eliminate_dead_code(
        self, is_impure_node: Optional[Callable[[Node], bool]] = None
    ) -> bool:
        """
        Remove all dead code from the graph, based on each node's number of
        users, and whether the nodes have any side effects. The graph must be
        topologically sorted before calling.

        Args:
            is_impure_node (Optional[Callable[[Node], bool]]): A function that returns
            whether a node is impure. If this is None, then the default behavior is to
            use Node.is_impure.

        Returns:
          bool: Whether the graph was changed as a result of the pass.

        Example:

        Before dead code is eliminated, `a` from `a = x + 1` below has no users
        and thus can be eliminated from the graph without having an effect.

        .. code-block:: python

            def forward(self, x):
                a = x + 1
                return x + self.attr_1

        After dead code is eliminated, `a = x + 1` has been removed, and the rest
        of `forward` remains.

        .. code-block:: python

            def forward(self, x):
                return x + self.attr_1

        .. warning::

            Dead code elimination has some heuristics to avoid removing
            side-effectful nodes (see Node.is_impure) but in general coverage
            is very bad, so you should assume that this method is not sound
            to call unless you know that your FX graph consists entirely
            of functional operations or you supply your own custom
            function for detecting side-effectful nodes.
        """
        from torch.utils._ordered_set import OrderedSet

        # Lint the graph first to make sure its topologically sorted, otherwise
        # DCE below will not behave as expected.
        self.lint()

        impure_random = True
        if torch._guards.TracingContext.try_get():
            impure_random = torch._inductor.config.fallback_random

        def has_side_effect(node):
            if is_impure_node is not None:
                return is_impure_node(node)
            return node.is_impure(impure_random)

        # Reverse iterate so that when we remove a node, any nodes used as an
        # input to that node have an updated user count that no longer reflects
        # the removed node.
        changed = False
        for node in reversed(self.nodes):
            if not has_side_effect(node) and len(node.users) == 0:
                self.erase_node(node)
                changed = True

        # Call DCE on the subgraphs
        if self.owning_module is not None:
            subgraph_names = OrderedSet(
                x.target for x in self.find_nodes(op="get_attr")
            )
            for child_name, child_module in self.owning_module.named_children():
                # Sometimes an owning_module can have unused children. Skip them
                # by checking them from get_attr node targets.
                if child_name in subgraph_names and isinstance(
                    child_module, torch.fx.GraphModule
                ):
                    changed |= child_module.graph.eliminate_dead_code()
                    child_module.recompile()

        return changed

    @compatibility(is_backward_compatible=False)
    def set_codegen(self, codegen: CodeGen):
        self._codegen = codegen

    @compatibility(is_backward_compatible=False)
    def on_generate_code(
        self,
        make_transformer: Callable[[Optional[TransformCodeFunc]], TransformCodeFunc],
    ):
        """Register a transformer function when python code is generated

        Args:
            make_transformer (Callable[[Optional[TransformCodeFunc]], TransformCodeFunc]):
                a function that returns a code transformer to be registered.
                This function is called by `on_generate_code` to obtain the
                code transformer.

                This function is also given as its input the currently
                registered code transformer (or None if nothing is registered),
                in case it is not desirable to overwrite it. This is useful to
                chain code transformers together.

        Returns:
            a context manager that when used in a `with` statement, to automatically
            restore the previously registered code transformer.

        Example:

        .. code-block:: python


            gm: fx.GraphModule = ...


            # This is a code transformer we want to register. This code
            # transformer prepends a pdb import and trace statement at the very
            # beginning of the generated torch.fx code to allow for manual
            # debugging with the PDB library.
            def insert_pdb(body):
                return ["import pdb; pdb.set_trace()\\n", *body]


            # Registers `insert_pdb`, and overwrites the current registered
            # code transformer (given by `_` to the lambda):
            gm.graph.on_generate_code(lambda _: insert_pdb)

            # Or alternatively, registers a code transformer which first
            # runs `body` through existing registered transformer, then
            # through `insert_pdb`:
            gm.graph.on_generate_code(
                lambda current_trans: (
                    lambda body: insert_pdb(
                        current_trans(body) if current_trans else body
                    )
                )
            )

            gm.recompile()
            gm(*inputs)  # drops into pdb


        This function can also be used as a context manager, with the benefit to
        automatically restores the previously registered code transformer:

        .. code-block:: python

            # ... continue from previous example

            with gm.graph.on_generate_code(lambda _: insert_pdb):
                # do more stuff with `gm`...
                gm.recompile()
                gm(*inputs)  # drops into pdb

            # now previous code transformer is restored (but `gm`'s code with pdb
            # remains - that means you can run `gm` with pdb here too, until you
            # run next `recompile()`).
        """
        on_gen_code_old = self._codegen._body_transformer
        self._codegen._body_transformer = make_transformer(on_gen_code_old)

        @contextlib.contextmanager
        def on_generate_code_context_manager():
            try:
                yield
            finally:
                self._codegen._body_transformer = on_gen_code_old

        return on_generate_code_context_manager()


@contextmanager
def _override_sym_repr(
    override: Callable[["torch.types.PySymType"], str],
) -> Iterator[None]:
    tmp = CodeGen._sym_repr
    try:
        CodeGen._sym_repr = override
        yield
    finally:
        CodeGen._sym_repr = tmp


def _identity(x):
    return x


def _make_color_fn(code):
    def f(s):
        reset = "\033[0m"
        return f"{code}{s}{reset}"

    return f


_color_codes = {
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "red": "\033[31m",
    "dim": "\033[2m",
    "dim_blue": "\033[2m\033[34m",
    "dim_green": "\033[2m\033[32m",
}
_color_fns = {k: _make_color_fn(v) for k, v in _color_codes.items()}
_counter_regexp = re.compile(r"# COUNTER: (\d+)")


reflectable_magic_methods = {
    "add": "{} + {}",
    "sub": "{} - {}",
    "mul": "{} * {}",
    "floordiv": "{} // {}",
    "truediv": "{} / {}",
    "div": "{} / {}",
    "mod": "{} % {}",
    "pow": "{} ** {}",
    "lshift": "{} << {}",
    "rshift": "{} >> {}",
    "and_": "{} & {}",
    "or_": "{} | {}",
    "xor": "{} ^ {}",
    "getitem": "{}[{}]",
    "matmul": "{} @ {}",
}

magic_methods = {
    "eq": "{} == {}",
    "ne": "{} != {}",
    "lt": "{} < {}",
    "gt": "{} > {}",
    "le": "{} <= {}",
    "ge": "{} >= {}",
    "pos": "+{}",
    "neg": "-{}",
    "invert": "~{}",
    **reflectable_magic_methods,
}

inplace_methods = {
    "iadd": "{} += {}",
    "iand": "{} &= {}",
    "ifloordiv": "{} //= {}",
    "ilshift": "{} <<= {}",
    "imod": "{} %= {}",
    "imul": "{} *= {}",
    "imatmul": "{} @= {}",
    "ior": "{} |= {}",
    "ipow": "{} **= {}",
    "irshift": "{} >>= {}",
    "isub": "{} -= {}",
    "itruediv": "{} /= {}",
    "ixor": "{} ^= {}",
    "setitem": "{}[{}] = {}",
}
