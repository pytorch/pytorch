# ruff: noqa: S101

import inspect
import itertools
import operator
import threading
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._library.opaque_object import (
    get_opaque_type_name,
    is_opaque_type,
    register_opaque_type,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_mode,
)


_trace_outputs: dict[int, list[Any]] = {}
_POSITIONAL_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}


class ArgSpecs:
    def __init__(self, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec | None):
        self.in_spec = in_spec
        self.out_spec = out_spec

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArgSpecs):
            return False
        return (self.in_spec, self.out_spec) == (other.in_spec, other.out_spec)

    def __hash__(self) -> int:
        return hash((self.in_spec, self.out_spec))

    def __fx_repr__(self) -> tuple[str, dict[str, type]]:
        in_dump = pytree.treespec_dumps(self.in_spec)
        out_spec = self.out_spec
        out_dump = None if out_spec is None else pytree.treespec_dumps(out_spec)
        return f"ArgSpecs.from_dumps({in_dump!r}, {out_dump!r})", {"ArgSpecs": ArgSpecs}

    @classmethod
    def from_dumps(cls, in_dump: str, out_dump: str | None) -> "ArgSpecs":
        out_spec = None if out_dump is None else pytree.treespec_loads(out_dump)
        return cls(pytree.treespec_loads(in_dump), out_spec)


register_opaque_type(ArgSpecs, typ="value")
ARG_SPECS_TYPE = get_opaque_type_name(ArgSpecs)


_INPUT_SCHEMA_TYPES = {
    torch.Tensor: "Tensor",
    type(None): "NoneType",
    bool: "bool",
    int: "SymInt",
    float: "float",
    str: "str",
    torch.dtype: "ScalarType",
    torch.device: "Device",
}
_OUTPUT_SCHEMA_TYPES = {
    torch.Tensor: "Tensor",
    int: "SymInt",
}


def schema_type(value: Any, *, is_output: bool = False) -> str:
    table = _OUTPUT_SCHEMA_TYPES if is_output else _INPUT_SCHEMA_TYPES
    for typ, schema in table.items():
        if isinstance(value, typ):
            return schema
    if is_opaque_type(type(value)):
        return get_opaque_type_name(type(value))
    raise TypeError(
        f"unsupported {'output' if is_output else 'input'} leaf "
        f"{value!r} of type {type(value).__name__}"
    )


def tensor_metadata(t: torch.Tensor) -> tuple[Any, ...]:
    strides = tuple(t.stride()) if t.layout is torch.strided else None
    offset = t.storage_offset() if t.layout is torch.strided else None
    return tuple(t.shape), strides, offset, t.dtype, t.device, t.layout


class AutoCustomOp:
    def __init__(
        self,
        qualname: str,
        fn: Callable[..., Any],
        mutates_args: Iterable[str] | str = (),
    ):
        self.ns, self.name = qualname.split("::")
        if "." in self.name:
            raise ValueError("auto_custom_op generates overload names")
        self.fn = fn
        self.sig = inspect.signature(fn)
        if mutates_args == "unknown":
            self.mutates_args = "unknown"
        else:
            self.mutates_args = frozenset(mutates_args)
        if self.mutates_args != "unknown":
            bad = self.mutates_args - self.sig.parameters.keys()
            if bad:
                raise ValueError(f"mutates_args names not in signature: {sorted(bad)}")
        self.lib = torch.library.Library(self.ns, "FRAGMENT")
        self.cache: dict[tuple[Any, ...], Any] = {}
        self.lock = threading.Lock()
        self.next_idx = itertools.count()

    def run_impl(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        with torch._C._AutoDispatchBelowADInplaceOrView():
            return self.fn(*args, **kwargs)

    def flat_arg_names(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        params = tuple(self.sig.parameters.values())
        positional = [p.name for p in params if p.kind in _POSITIONAL_KINDS]
        vararg = None
        for p in params:
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                vararg = p.name

        def name_trees(name: str, value: Any) -> tuple[Any, Any]:
            leaves, spec = pytree.tree_flatten(value)
            if spec.is_leaf():
                schema_names = [name]
            else:
                schema_names = [f"{name}{i}" for i in range(len(leaves))]
            top = pytree.tree_unflatten([name] * len(leaves), spec)
            schema = pytree.tree_unflatten(schema_names, spec)
            return top, schema

        top_args = []
        schema_args = []
        for i, arg in enumerate(args):
            name = positional[i] if i < len(positional) else vararg or f"arg{i}"
            top, schema = name_trees(name, arg)
            top_args.append(top)
            schema_args.append(schema)
        top_kwargs, schema_kwargs = {}, {}
        for key, value in kwargs.items():
            top_kwargs[key], schema_kwargs[key] = name_trees(key, value)

        used = {"arg_specs"}
        schema_names = []
        for name in pytree.tree_leaves((schema_args, schema_kwargs)):
            count = 0
            unique = name
            while unique in used:
                count += 1
                unique = f"{name}_{count}"
            used.add(unique)
            schema_names.append(unique)

        return pytree.tree_leaves((top_args, top_kwargs)), schema_names

    def schema_annotations(
        self,
        flat_in: list[Any],
        flat_out: list[Any],
        mutated: set[int],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        leaves = []
        for kind, values in (("in", flat_in), ("out", flat_out)):
            for i, x in enumerate(values):
                if isinstance(x, torch.Tensor):
                    leaves.append((kind, i, x))

        groups = []
        for _, _, t in leaves:
            group = None
            for j, (_, _, prev) in enumerate(leaves[: len(groups)]):
                if torch._C._is_alias_of(t, prev):
                    group = groups[j]
                    break
            groups.append(len(groups) if group is None else group)

        in_by_group: dict[int, list[int]] = {}
        out_by_group: dict[int, list[int]] = {}
        for group, (kind, i, _) in zip(groups, leaves, strict=True):
            by_group = in_by_group if kind == "in" else out_by_group
            by_group.setdefault(group, []).append(i)

        mutated_groups = set()
        for group, indices in in_by_group.items():
            if any(i in mutated for i in indices):
                mutated_groups.add(group)

        annotated_groups = set()
        for group in set(groups):
            multiple_inputs = len(in_by_group.get(group, ())) > 1
            multiple_outputs = len(out_by_group.get(group, ())) > 1
            input_output_alias = in_by_group.get(group) and out_by_group.get(group)
            if multiple_inputs or multiple_outputs or input_output_alias:
                annotated_groups.add(group)
            elif group in mutated_groups:
                annotated_groups.add(group)
        names = {group: f"a{i}" for i, group in enumerate(sorted(annotated_groups))}
        in_group = {}
        for group, indices in in_by_group.items():
            for i in indices:
                in_group[i] = group

        out_group = {}
        for group, indices in out_by_group.items():
            for i in indices:
                out_group[i] = group

        in_schema = []
        for i, x in enumerate(flat_in):
            ann = ""
            group = in_group.get(i)
            if group in annotated_groups:
                ann = f"({names[group]}{'!' if i in mutated else ''})"
            in_schema.append(f"{schema_type(x)}{ann}")

        out_schema = []
        for i, x in enumerate(flat_out):
            ann = ""
            group = out_group.get(i)
            if group in annotated_groups:
                ann = f"({names[group]}{'!' if group in mutated_groups else ''})"
            out_schema.append(f"{schema_type(x, is_output=True)}{ann}")

        return tuple(in_schema), tuple(out_schema)

    def op_for_schema(
        self,
        in_schema: tuple[str, ...],
        out_schema: tuple[str, ...],
        arg_names: list[str],
    ) -> Any:
        key = (in_schema, out_schema, tuple(arg_names))
        op = self.cache.get(key)
        if op is not None:
            return op

        with self.lock:
            op = self.cache.get(key)
            if op is not None:
                return op

            idx = next(self.next_idx)
            arg_decls = [f"{s} {n}" for s, n in zip(in_schema, arg_names)]
            args_decl = ", ".join([*arg_decls, f"*, {ARG_SPECS_TYPE} arg_specs"])
            if len(out_schema) == 0:
                ret_decl = "()"
            elif len(out_schema) == 1:
                ret_decl = out_schema[0]
            else:
                ret_decl = f"({', '.join(out_schema)})"
            schema = f"{self.name}.gen{idx}({args_decl}) -> {ret_decl}"
            self.lib.define(schema)

            def kernel(*flat_args: Any, arg_specs: ArgSpecs) -> Any:
                cached = _trace_outputs.pop(id(arg_specs), None)
                if cached is not None:
                    if len(cached) == 0:
                        return None
                    return cached[0] if len(cached) == 1 else tuple(cached)

                args, kwargs = pytree.tree_unflatten(flat_args, arg_specs.in_spec)
                out = self.run_impl(args, kwargs)
                if arg_specs.out_spec is None:
                    if out is not None:
                        raise RuntimeError("expected output structure None")
                    return None
                flat_out, out_spec = pytree.tree_flatten(out)
                if out_spec != arg_specs.out_spec:
                    raise RuntimeError("output pytree structure changed")
                if len(flat_out) == 0:
                    return None
                return flat_out[0] if len(flat_out) == 1 else tuple(flat_out)

            qualname = f"{self.name}.gen{idx}"
            self.lib.impl(qualname, kernel, "CompositeExplicitAutograd")
            packet = getattr(getattr(torch.ops, self.ns), self.name)
            op = getattr(packet, f"gen{idx}")
            self.cache[key] = op
            return op

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        mode = get_proxy_mode()
        if mode is None:
            return self.run_impl(args, kwargs)
        if mode.tracing_mode != "real":
            raise RuntimeError("auto_custom_op only supports tracing_mode='real'")

        flat_in, in_spec = pytree.tree_flatten((args, kwargs))
        flat_names, arg_names = self.flat_arg_names(args, kwargs)
        mutates_all = self.mutates_args == "unknown"
        mutated = set()
        for i, x in enumerate(flat_in):
            if not isinstance(x, torch.Tensor):
                continue
            if mutates_all or flat_names[i] in self.mutates_args:
                mutated.add(i)

        metadata = []
        for x in flat_in:
            metadata.append(tensor_metadata(x) if isinstance(x, torch.Tensor) else None)

        with disable_proxy_modes_tracing():
            out = self.fn(*args, **kwargs)

        for i, x in enumerate(flat_in):
            if isinstance(x, torch.Tensor) and tensor_metadata(x) != metadata[i]:
                raise RuntimeError("metadata-changing input mutations are unsupported")

        if out is None:
            flat_out, out_spec = [], None
        else:
            flat_out, out_spec = pytree.tree_flatten(out)

        in_schema, out_schema = self.schema_annotations(flat_in, flat_out, mutated)
        op = self.op_for_schema(in_schema, out_schema, arg_names)
        arg_specs = ArgSpecs(in_spec, out_spec)

        _trace_outputs[id(arg_specs)] = flat_out
        try:
            with disable_proxy_modes_tracing():
                call_kwargs = {"arg_specs": arg_specs}
                packed = mode.__torch_dispatch__(op, (), tuple(flat_in), call_kwargs)
        finally:
            _trace_outputs.pop(id(arg_specs), None)

        if out_spec is None:
            return None

        op_node = None
        for node in reversed(mode.tracer.graph.nodes):
            if node.op == "call_function" and node.target is op:
                op_node = node
                break
        assert op_node is not None

        op_proxy = mode.tracer.proxy(op_node)
        getitems = {}
        for node in mode.tracer.graph.nodes:
            if node.op != "call_function" or node.target is not operator.getitem:
                continue
            if node.args[0] is op_node:
                getitems[node.args[1]] = mode.tracer.proxy(node)

        if len(flat_out) == 1:
            packed = (packed,)
        proxy_out = []
        for i, x in enumerate(packed):
            if isinstance(x, torch.Tensor):
                proxy_out.append(x)
            elif len(flat_out) == 1:
                proxy_out.append(op_proxy)
            elif i in getitems:
                proxy_out.append(getitems[i])
            else:
                proxy_out.append(op_proxy[i])
        return pytree.tree_unflatten(proxy_out, out_spec)


def auto_custom_op(
    qualname: str,
    *,
    mutates_args: Iterable[str] | str = (),
) -> Callable[[Callable[..., Any]], AutoCustomOp]:
    """Decorate a Python function as a custom operator packet.

    Eager calls run the Python function below ADInplaceOrView. During
    ``make_fx(..., tracing_mode="real")``, each observed behavior gets a
    generated overload, ``torch.ops.<namespace>.<name>.genN``. The overload
    schema is inferred from that call's flattened inputs, flattened outputs,
    aliasing, and declared mutations.

    Inputs and outputs may be pytrees or registered opaque objects. Pytree
    leaves become flat operator arguments/results; the pytree specs are carried
    by a final keyword-only ``arg_specs`` opaque argument. Supported input leaf
    types are ``Tensor``, ``None``, ``bool``, ``SymInt``/``int``, ``float``,
    ``str``, ``dtype``, ``device``, and registered opaque objects. Supported
    output leaf types are ``Tensor``, ``SymInt``/``int``, and registered opaque
    objects.

    Arbitrary Python objects must be converted by the user into either pytrees
    of supported leaves or registered opaque objects. ``mutates_args`` declares
    top-level Python arguments that may mutate tensor leaves and controls the
    generated schema's mutation annotations.
    """

    return lambda fn: AutoCustomOp(qualname, fn, mutates_args)


if __name__ == "__main__":
    import re

    import expecttest

    from torch.fx.experimental.proxy_tensor import make_fx

    NS = "auto_op"

    def schema(op):
        return str(next(iter(op.cache.values()))._schema)

    def graph_arg(arg):
        return arg.name if hasattr(arg, "name") else arg

    def graph_code(gm):
        arg_specs_re = r"arg_specs = ArgSpecs\.from_dumps\('[^']*', (?:'[^']*'|None)\)"
        return re.sub(arg_specs_re, "arg_specs=...", gm.code.strip())

    # Basic trace: pytree outputs, alias annotations, and declared mutations.
    @auto_custom_op(f"{NS}::viewy")
    def viewy(t):
        return t.view(-1), t * 2

    @auto_custom_op(f"{NS}::mutator", mutates_args=("tensors",))
    def mutator(tensors, val):
        for t in tensors:
            t.add_(val)
        return tensors[0].view(-1)

    @auto_custom_op(f"{NS}::twin_outs")
    def twin_outs(t):
        y = t * 2
        return y, y.view(-1)

    def f(a, b):
        v, doubled = viewy(a)
        mv = mutator([a, b], 1.0)
        o1, o2 = twin_outs(b)
        return v + doubled[0] + mv + o1 + o2

    a, b = torch.randn(3), torch.randn(3)
    gm = make_fx(f, tracing_mode="real")(a.clone(), b.clone())
    custom_ops = set(viewy.cache.values())
    custom_ops.update(mutator.cache.values())
    custom_ops.update(twin_outs.cache.values())
    custom_calls = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in custom_ops:
            args = tuple(map(graph_arg, node.args))
            custom_calls.append((node.name, str(node.target), args, tuple(node.kwargs)))
    assert custom_calls == [
        ("viewy", f"{NS}.viewy.gen0", ("a_1",), ("arg_specs",)),
        ("mutator", f"{NS}.mutator.gen0", ("a_1", "b_1", 1.0), ("arg_specs",)),
        ("twin_outs", f"{NS}.twin_outs.gen0", ("b_1",), ("arg_specs",)),
    ]
    expecttest.TestCase().assertExpectedInline(
        graph_code(gm),
        f"""\
def forward(self, a_1, b_1):
    viewy = torch.ops.{NS}.viewy.gen0(a_1, arg_specs=...)
    getitem = viewy[0]
    getitem_1 = viewy[1];  viewy = None
    mutator = torch.ops.{NS}.mutator.gen0(a_1, b_1, 1.0, arg_specs=...);  a_1 = None
    twin_outs = torch.ops.{NS}.twin_outs.gen0(b_1, arg_specs=...);  b_1 = None
    getitem_2 = twin_outs[0]
    getitem_3 = twin_outs[1];  twin_outs = None
    select = torch.ops.aten.select.int(getitem_1, 0, 0);  getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem, select);  getitem = select = None
    add_1 = torch.ops.aten.add.Tensor(add, mutator);  add = mutator = None
    add_2 = torch.ops.aten.add.Tensor(add_1, getitem_2);  add_1 = getitem_2 = None
    add_3 = torch.ops.aten.add.Tensor(add_2, getitem_3);  add_2 = getitem_3 = None
    return add_3""",
    )

    print("\n=== Generated operator schemas ===")
    for op_obj in (viewy, mutator, twin_outs):
        for op in op_obj.cache.values():
            print(op._schema)

    print("\n=== Graph ===")
    print(gm.code)

    a1, b1 = a.clone(), b.clone()
    replay = gm(a1, b1)
    a2, b2 = a.clone(), b.clone()
    eager = f(a2, b2)
    print("replay == eager:", torch.allclose(replay, eager))
    mutated_a = torch.allclose(a1, a2)
    mutated_b = torch.allclose(b1, b2)
    print("mutation side effect matches:", mutated_a, mutated_b)

    # Pytree input leaves are named after the top-level argument plus an index.
    assert schema(viewy) == (
        f"{NS}::viewy.gen0(Tensor(a0) t, *, PyObject arg_specs) -> (Tensor(a0), Tensor)"
    )
    assert schema(mutator) == (
        f"{NS}::mutator.gen0(Tensor(a0!) tensors0, Tensor(a1!) tensors1, "
        f"float val, *, PyObject arg_specs) -> Tensor(a0!)"
    )
    assert schema(twin_outs) == (
        f"{NS}::twin_outs.gen0(Tensor t, *, PyObject arg_specs) "
        f"-> (Tensor(a0), Tensor(a0))"
    )

    # Schema argument names come from positional, pytree, and keyword arguments.
    @auto_custom_op(f"{NS}::named_args")
    def named_args(x, pair, *, scale):
        return x + pair[0] + pair[1] + scale

    make_fx(lambda x, y, z: named_args(x, (y, z), scale=2))(a, b, torch.randn(3))
    named_schema = schema(named_args)
    print("named args schema:", named_schema)
    assert named_schema == (
        f"{NS}::named_args.gen0(Tensor x, Tensor pair0, Tensor pair1, "
        f"SymInt scale, *, PyObject arg_specs) -> Tensor"
    )

    count = [0]

    # make_fx tracing invokes the wrapped function once and emits one custom op.
    @auto_custom_op(f"{NS}::counted")
    def counted(x):
        count[0] += 1
        return x + 1

    counted_gm = make_fx(lambda x: counted(x), tracing_mode="real")(a)
    assert count[0] == 1
    counted_calls = []
    for node in counted_gm.graph.nodes:
        if node.op == "call_function" and node.target in counted.cache.values():
            counted_calls.append(str(node.target))
    assert counted_calls == [f"{NS}.counted.gen0"]
    assert torch.allclose(counted_gm(a), a + 1)
    assert count[0] == 2

    # Data mutations are not inferred; mutates_args controls schema annotations.
    @auto_custom_op(f"{NS}::undeclared")
    def undeclared(x):
        x.add_(1)
        return x

    make_fx(lambda x: undeclared(x))(torch.randn(3))
    assert schema(undeclared) == (
        f"{NS}::undeclared.gen0(Tensor(a0) x, *, PyObject arg_specs) -> Tensor(a0)"
    )

    # Metadata-changing tensor mutations are rejected.
    @auto_custom_op(f"{NS}::metadata_mutation", mutates_args=("x",))
    def metadata_mutation(x):
        x.t_()
        return x

    try:
        make_fx(lambda x: metadata_mutation(x))(torch.randn(2, 3))
    except RuntimeError as e:
        assert "metadata-changing input mutations" in str(e)
    else:
        raise AssertionError("metadata-changing mutation should fail")

    # Only real-mode make_fx tracing is supported by this prototype.
    @auto_custom_op(f"{NS}::fake")
    def fake_reject(x):
        return x + 1

    try:
        make_fx(lambda x: fake_reject(x), tracing_mode="fake")(torch.randn(3))
    except RuntimeError as e:
        assert "tracing_mode='real'" in str(e)
    else:
        raise AssertionError("fake tracing should fail")

    # None is a valid input leaf and appears as NoneType in the generated schema.
    @auto_custom_op(f"{NS}::none_input")
    def none_input(x, maybe_bias):
        return x if maybe_bias is None else x + maybe_bias

    none_gm = make_fx(lambda x: none_input(x, None))(torch.randn(3))
    assert torch.allclose(none_gm(a), a)
    none_schema = schema(none_input)
    print("none input schema:", none_schema)
    assert none_schema == (
        f"{NS}::none_input.gen0(Tensor(a0) x, NoneType maybe_bias, "
        f"*, PyObject arg_specs) -> Tensor(a0)"
    )

    # Different observed calling conventions generate distinct overloads.
    @auto_custom_op(f"{NS}::multiple_overloads")
    def multiple_overloads(x, values):
        if values is None:
            return x + 1
        return x + sum(values)

    make_fx(lambda x: multiple_overloads(x, None))(torch.randn(3))
    make_fx(lambda x: multiple_overloads(x, [1, 2]))(torch.randn(3))
    overload_schemas = [str(op._schema) for op in multiple_overloads.cache.values()]
    print("multiple overload schemas:")
    for overload_schema in overload_schemas:
        print(overload_schema)
    assert overload_schemas == [
        f"{NS}::multiple_overloads.gen0("
        f"Tensor x, NoneType values, *, PyObject arg_specs) -> Tensor",
        f"{NS}::multiple_overloads.gen1(Tensor x, SymInt values0, "
        f"SymInt values1, *, PyObject arg_specs) -> Tensor",
    ]

    # Alias annotation names are numeric, so more than 26 alias groups works.
    @auto_custom_op(f"{NS}::many_alias_groups")
    def many_alias_groups(xs):
        return tuple(x.view(-1) for x in xs)

    many = [torch.randn(2) for _ in range(30)]
    make_fx(lambda *xs: many_alias_groups(list(xs)), tracing_mode="real")(*many)
    many_inputs = ", ".join(f"Tensor(a{i}) xs{i}" for i in range(30))
    many_outputs = ", ".join(f"Tensor(a{i})" for i in range(30))
    assert schema(many_alias_groups) == (
        f"{NS}::many_alias_groups.gen0({many_inputs}, "
        f"*, PyObject arg_specs) -> ({many_outputs})"
    )

    # SymInt outputs are valid schema results.
    @auto_custom_op(f"{NS}::symint_output")
    def symint_output(x):
        return x.shape[0]

    symint_gm = make_fx(lambda x: symint_output(x))(torch.randn(3))
    assert symint_gm(torch.randn(3)) == 3
    symint_schema = schema(symint_output)
    print("symint output schema:", symint_schema)
    assert symint_schema == (
        f"{NS}::symint_output.gen0(Tensor x, *, PyObject arg_specs) -> SymInt"
    )

    # Registered opaque objects are valid input and output leaves.
    class Scale:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Scale) and self.value == other.value

        def __hash__(self):
            return hash(self.value)

        def __fx_repr__(self):
            return f"Scale({self.value!r})", {"Scale": Scale}

    register_opaque_type(Scale, typ="value")

    @auto_custom_op(f"{NS}::opaque_io")
    def opaque_io(x, scale):
        return x * scale.value, scale

    opaque_gm = make_fx(lambda x: opaque_io(x, Scale(3)), tracing_mode="real")(a)
    opaque_out, opaque_scale = opaque_gm(a)
    opaque_schema = schema(opaque_io)
    print("opaque schema:", opaque_schema)
    assert torch.allclose(opaque_out, a * 3)
    assert opaque_scale == Scale(3)
    assert opaque_schema == (
        f"{NS}::opaque_io.gen0("
        f"Tensor x, PyObject scale, *, PyObject arg_specs) -> (Tensor, PyObject)"
    )

    print("auto_custom_op.py tests passed")
