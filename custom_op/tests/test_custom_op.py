# ruff: noqa: S101
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.utils._pytree as pytree
from torch._library.opaque_object import register_opaque_type
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# tests/ is not a package; make the repo-root custom_op package importable when
# this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from custom_op import custom_op
from custom_op.dispatchless_op_overload import DispatchlessOpOverloadPacket
from custom_op.schema import decode_overload_name, encode_overload_name


def call_nodes(gm, op):
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target in op._cache.values()
    ]


def debug_overload_name(op):
    """Return the generated overload name a custom_op overload carries."""
    if not isinstance(op, torch._ops.OpOverload):
        raise TypeError("expected an OpOverload")
    name = op.__dict__.get("_custom_op_overload_name")
    if name is None:
        raise TypeError("expected a custom_op overload")
    return name


def debug_overload(op):
    return debug_overload_name(next(iter(op._cache.values())))


# Fixtures are defined at module scope so their __qualname__ is a plain name that
# FX can codegen (a class nested in a method has a "<locals>" qualname).
@dataclass
class Box:
    tensor: torch.Tensor


pytree.register_dataclass(Box, serialized_type_name="custom_op.Box")


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


@instantiate_parametrized_tests
class TestCustomOp(TestCase):
    def test_basic_trace(self):
        # pytree outputs, alias annotations, and declared mutations in one graph.
        @custom_op("auto_op::viewy")
        def viewy(t):
            return t.view(-1), t * 2

        @custom_op("auto_op::mutator", mutates_args=("tensors",))
        def mutator(tensors, val):
            for t in tensors:
                t.add_(val)
            return tensors[0].view(-1)

        @custom_op("auto_op::twin_outs")
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
        assert all(
            isinstance(call_nodes(gm, op)[0].target, torch._ops.OpOverload)
            for op in (viewy, mutator, twin_outs)
        )
        # Node args are FLAT leaves matching the schema; the structured pytree
        # form is shown only in printouts (see the assertExpectedInline blocks).
        assert call_nodes(gm, viewy)[0].args[0].name == "a_1"
        assert [n.name for n in call_nodes(gm, mutator)[0].args[:2]] == ["a_1", "b_1"]
        assert call_nodes(gm, mutator)[0].args[2] == 1.0
        assert call_nodes(gm, twin_outs)[0].args[0].name == "b_1"
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %a_1 : [num_users=2] = placeholder[target=a_1]
    %b_1 : [num_users=2] = placeholder[target=b_1]
    %viewy : [num_users=2] = call_function[target=torch.ops.auto_op.viewy[*]](args = (%a_1,), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%viewy, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%viewy, 1), kwargs = {})
    %mutator : [num_users=1] = call_function[target=torch.ops.auto_op.mutator[*]](args = ([%a_1, %b_1], 1.0), kwargs = {})
    %twin_outs : [num_users=2] = call_function[target=torch.ops.auto_op.twin_outs[*]](args = (%b_1,), kwargs = {})
    %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%twin_outs, 0), kwargs = {})
    %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%twin_outs, 1), kwargs = {})
    %select : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_1, 0, 0), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, %select), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mutator), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %getitem_2), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %getitem_3), kwargs = {})
    return add_3""",
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, a_1, b_1):
    viewy = torch.ops.auto_op.viewy(a_1)
    getitem = viewy[0]
    getitem_1 = viewy[1];  viewy = None
    mutator = torch.ops.auto_op.mutator([a_1, b_1], 1.0);  a_1 = None
    twin_outs = torch.ops.auto_op.twin_outs(b_1);  b_1 = None
    getitem_2 = twin_outs[0]
    getitem_3 = twin_outs[1];  twin_outs = None
    select = torch.ops.aten.select.int(getitem_1, 0, 0);  getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem, select);  getitem = select = None
    add_1 = torch.ops.aten.add.Tensor(add, mutator);  add = mutator = None
    add_2 = torch.ops.aten.add.Tensor(add_1, getitem_2);  add_1 = getitem_2 = None
    add_3 = torch.ops.aten.add.Tensor(add_2, getitem_3);  add_2 = getitem_3 = None
    return add_3""",
        )

        # Generated code replays through the packet and reproduces mutations.
        a1, b1 = a.clone(), b.clone()
        replay = gm(a1, b1)
        a2, b2 = a.clone(), b.clone()
        eager = f(a2, b2)
        self.assertEqual(replay, eager)
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)

    def test_overload_identity(self):
        @custom_op("auto_op::ident")
        def ident(t):
            return t.view(-1), t * 2

        a = torch.randn(3)
        make_fx(lambda t: ident(t), tracing_mode="real")(a)
        assert debug_overload(ident).startswith("pt_")
        # The overload name is a reversible encoding of the metadata.
        meta = decode_overload_name(debug_overload(ident))
        assert meta["schema"] == "(Tensor(a0) t) -> (Tensor(a0), Tensor)"
        assert encode_overload_name(meta) == debug_overload(ident)
        # The packet lives at torch.ops.<ns>.<name>; calling it runs the fn.
        assert isinstance(torch.ops.auto_op.ident, DispatchlessOpOverloadPacket)
        y, doubled = torch.ops.auto_op.ident(a)
        assert torch.allclose(y, a.view(-1)) and torch.allclose(doubled, a * 2)
        # The overload has a normal OpOverload name/identity and is callable
        # directly.
        op = next(iter(ident._cache.values()))
        assert op.name() == "auto_op::ident"
        assert op._overloadname == debug_overload(ident)
        oy, od = op(a)
        assert torch.allclose(oy, a.view(-1)) and torch.allclose(od, a * 2)
        # Materialized (in-use) overloads are accessible through the packet by name,
        # and iterating the packet yields them.
        assert torch.ops.auto_op.ident.overloads() == [debug_overload(ident)]
        assert list(torch.ops.auto_op.ident) == [debug_overload(ident)]
        assert getattr(torch.ops.auto_op.ident, debug_overload(ident)) is op

    def test_overload_from_name(self):
        @custom_op("auto_op::rebuildable")
        def rebuildable(x):
            return x + 1

        make_fx(lambda x: rebuildable(x), tracing_mode="real")(torch.randn(3))
        packet = torch.ops.auto_op.rebuildable
        op = next(iter(rebuildable._cache.values()))
        name = debug_overload(rebuildable)

        # Drop the cache: the name is a reversible encoding, so querying it decodes
        # + re-registers the overload for a convention that isn't materialized.
        rebuildable._cache.clear()
        assert packet.overloads() == []
        rebuilt = getattr(packet, name)
        assert rebuilt._overloadname == name
        assert str(rebuilt._schema) == str(op._schema)
        # It is now materialized (registered) again.
        assert packet.overloads() == [name]

        # Non-overload / malformed names raise AttributeError (getattr-friendly).
        with self.assertRaises(AttributeError):
            packet.not_an_overload
        with self.assertRaises(AttributeError):
            packet.pt_garbage

    def test_pytree_args(self):
        a, b, c = torch.randn(3), torch.randn(3), torch.randn(3)

        # Positional, keyword-only, and tuple-nested args; flat leaves in the
        # node, structured pytree only in the printout.
        @custom_op("auto_op::named_args")
        def named_args(x, pair, *, scale):
            return x + pair[0] + pair[1] + scale

        named_gm = make_fx(lambda x, y, z: named_args(x, (y, z), scale=2))(a, b, c)
        named_node = call_nodes(named_gm, named_args)[0]
        assert [n.name for n in named_node.args[:3]] == ["x_1", "y_1", "z_1"]
        assert named_node.args[3] == 2
        assert (
            "torch.ops.auto_op.named_args(x_1, (y_1, z_1), scale = 2)" in named_gm.code
        )

        @custom_op("auto_op::nested_arg")
        def nested_arg(values):
            return values[0][0] + values[0][1] + values[1]

        nested_gm = make_fx(
            lambda x, y, z: nested_arg([[x, y], z]), tracing_mode="real"
        )(a, b, c)
        assert "target=torch.ops.auto_op.nested_arg[*]" in str(nested_gm.graph)
        assert "args = ([[%x_1, %y_1], %z_1],)" in str(nested_gm.graph)
        assert "torch.ops.auto_op.nested_arg([[x_1, y_1], z_1])" in nested_gm.code

    def test_traces_fn_once(self):
        # make_fx tracing invokes the wrapped function once and emits one op.
        count = [0]

        @custom_op("auto_op::counted")
        def counted(x):
            count[0] += 1
            return x + 1

        a = torch.randn(3)
        counted_gm = make_fx(lambda x: counted(x), tracing_mode="real")(a)
        assert count[0] == 1
        assert len(call_nodes(counted_gm, counted)) == 1
        assert torch.allclose(counted_gm(a), a + 1)
        assert count[0] == 2

    def test_undeclared_mutation_not_annotated(self):
        # Data mutations are not inferred; mutates_args controls annotations.
        @custom_op("auto_op::undeclared")
        def undeclared(x):
            x.add_(1)
            return x

        make_fx(lambda x: undeclared(x))(torch.randn(3))
        schema = next(iter(undeclared._cache.values()))._schema
        assert schema.name == "auto_op::undeclared"
        assert "!" not in str(schema)  # no mutation annotation

    def test_declared_path_mutation(self):
        @custom_op(
            "auto_op::path_mutator",
            mutates_args=("state['buf']", "buffers[0]", "box.tensor"),
        )
        def path_mutator(state, buffers, box):
            state["buf"].add_(1)
            buffers[0].add_(1)
            box.tensor.add_(1)
            return state["buf"].view(-1), buffers[1], box.tensor.view(-1)

        state = {"buf": torch.randn(3), "other": torch.randn(3)}
        buffers = [torch.randn(3), torch.randn(3)]
        box = Box(torch.randn(3))
        pm_gm = make_fx(lambda state, buffers, box: path_mutator(state, buffers, box))(
            state, buffers, box
        )
        path_schema = str(next(iter(path_mutator._cache.values()))._schema)
        assert "Tensor(a0!) state0" in path_schema
        assert "Tensor state1" in path_schema
        assert "Tensor(a1!) buffers0" in path_schema
        assert "Tensor(a2) buffers1" in path_schema
        assert "Tensor(a3!) box0" in path_schema
        # The dataclass reconstructs in the printout, and (via custom-builtin
        # registration of Box) the generated code replays.
        assert "box.tensor" not in pm_gm.code and "Box(tensor=box_1)" in pm_gm.code
        state2 = {"buf": torch.zeros(3), "other": torch.zeros(3)}
        buffers2 = [torch.zeros(3), torch.zeros(3)]
        box2 = Box(torch.zeros(3))
        pm_gm(state2, buffers2, box2)
        assert torch.allclose(box2.tensor, torch.ones(3))

    def test_unknown_mutation(self):
        @custom_op("auto_op::unknown_mutator", mutates_args="unknown")
        def unknown_mutator(x):
            x.add_(1)
            return x.view(-1)

        make_fx(lambda x: unknown_mutator(x))(torch.randn(3))
        assert "Tensor(a0!) x" in str(
            next(iter(unknown_mutator._cache.values()))._schema
        )

    def test_metadata_change_rejected(self):
        @custom_op("auto_op::metadata_mutation", mutates_args=("x",))
        def metadata_mutation(x):
            x.t_()
            return x

        with self.assertRaisesRegex(RuntimeError, "metadata-changing input mutations"):
            make_fx(lambda x: metadata_mutation(x))(torch.randn(2, 3))

    @parametrize("tracing_mode", ["fake", "symbolic"])
    def test_requires_fake_kernel(self, tracing_mode):
        # fake/symbolic tracing requires a registered fake kernel -- there is no
        # fallback to running the real fn under fake tensors.
        @custom_op("auto_op::no_fake")
        def no_fake(x):
            return x + 1

        with self.assertRaisesRegex(RuntimeError, "register_fake"):
            make_fx(lambda x: no_fake(x), tracing_mode=tracing_mode)(torch.randn(3))

    def test_register_fake(self):
        # When the real fn cannot run under fake tensors (data-dependent .item()
        # here), register_fake supplies output metadata for tracing_mode="fake".
        @custom_op("auto_op::needs_fake")
        def needs_fake(x):
            return x.new_full((x.numel(),), x.max().item())

        @needs_fake.register_fake
        def _needs_fake_fake(x):
            return x.new_empty((x.numel(),))

        nf_gm = make_fx(lambda x: needs_fake(x), tracing_mode="fake")(torch.randn(3))
        assert "torch.ops.auto_op.needs_fake(x_1)" in nf_gm.code
        assert call_nodes(nf_gm, needs_fake)[0].meta["val"].shape == (3,)
        # tracing_mode="real" runs the real fn (data-dependent path included).
        assert torch.allclose(nf_gm(torch.arange(3.0)), torch.full((3,), 2.0))

    def test_none_input(self):
        # None is a valid input leaf and appears as NoneType in the schema.
        @custom_op("auto_op::none_input")
        def none_input(x, maybe_bias):
            return x if maybe_bias is None else x + maybe_bias

        a = torch.randn(3)
        none_gm = make_fx(lambda x: none_input(x, None))(a)
        assert torch.allclose(none_gm(a), a)
        assert call_nodes(none_gm, none_input)[0].args[1] is None
        assert "torch.ops.auto_op.none_input(x_1, None)" in none_gm.code

    def test_divergent_overloads(self):
        # Different observed calling conventions generate distinct overloads,
        # but both still codegen through the packet name.
        @custom_op("auto_op::multiple_overloads")
        def multiple_overloads(x, values):
            if values is None:
                return x + 1
            return x + sum(values)

        none_gm = make_fx(lambda x: multiple_overloads(x, None))(torch.randn(3))
        list_gm = make_fx(lambda x: multiple_overloads(x, [1, 2]))(torch.randn(3))
        names = [debug_overload_name(op) for op in multiple_overloads._cache.values()]
        assert names[0] != names[1]
        assert "torch.ops.auto_op.multiple_overloads(x_1, None)" in none_gm.code
        assert "torch.ops.auto_op.multiple_overloads(x_1, [1, 2])" in list_gm.code

    def test_optional_tensor(self):
        @custom_op("auto_op::optional_tensor")
        def optional_tensor(x, maybe_bias):
            return x if maybe_bias is None else x + maybe_bias

        a, b = torch.randn(3), torch.randn(3)
        none_gm = make_fx(lambda x: optional_tensor(x, None), tracing_mode="real")(a)
        tensor_gm = make_fx(lambda x, y: optional_tensor(x, y), tracing_mode="real")(
            a, b
        )
        names = [debug_overload_name(op) for op in optional_tensor._cache.values()]
        assert names[0] != names[1]
        assert call_nodes(none_gm, optional_tensor)[0].args[1] is None
        assert call_nodes(tensor_gm, optional_tensor)[0].args[1].name == "y_1"

    def test_dict_input(self):
        @custom_op("auto_op::dict_values")
        def dict_values(values):
            return sum(values.values())

        a, b = torch.randn(3), torch.randn(3)
        one_gm = make_fx(lambda x: dict_values({"x": x}), tracing_mode="real")(a)
        two_gm = make_fx(
            lambda x, y: dict_values({"x": x, "y": y}), tracing_mode="real"
        )(a, b)
        names = [debug_overload_name(op) for op in dict_values._cache.values()]
        assert names[0] != names[1]
        assert "dict_values({'x': x_1})" in one_gm.code
        assert "dict_values({'x': x_1, 'y': y_1})" in two_gm.code

    def test_many_alias_groups(self):
        # Alias annotation names are numeric, so more than 26 groups works.
        @custom_op("auto_op::many_alias_groups")
        def many_alias_groups(xs):
            return tuple(x.view(-1) for x in xs)

        many = [torch.randn(2) for _ in range(30)]
        make_fx(lambda *xs: many_alias_groups(list(xs)), tracing_mode="real")(*many)
        schema = str(next(iter(many_alias_groups._cache.values()))._schema)
        assert all(f"Tensor(a{i}) xs{i}" in schema for i in range(30))

    def test_symint_output(self):
        @custom_op("auto_op::symint_output")
        def symint_output(x):
            return x.shape[0]

        symint_gm = make_fx(lambda x: symint_output(x))(torch.randn(3))
        assert symint_gm(torch.randn(3)) == 3
        assert (
            next(iter(symint_output._cache.values()))._schema.name
            == "auto_op::symint_output"
        )

    def test_opaque_object_io(self):
        # Registered opaque objects (module-level Scale) are valid I/O leaves.
        @custom_op("auto_op::opaque_io")
        def opaque_io(x, scale):
            return x * scale.value, scale

        a = torch.randn(3)
        gm = make_fx(lambda x: opaque_io(x, Scale(3)), tracing_mode="real")(a)
        out, scale = gm(a)
        assert torch.allclose(out, a * 3)
        assert scale == Scale(3)

    def test_exceeds_dispatcher_arg_limit(self):
        # Dispatch-less ops are constructed directly in Python, so they are not
        # bound by the dispatcher's 64-argument limit. A registered op would
        # fail at define() with more than 64 arguments.
        @custom_op("auto_op::many_inputs")
        def many_inputs(xs):
            return sum(xs[1:], xs[0].clone())

        big = [torch.randn(3) for _ in range(200)]
        gm = make_fx(lambda *xs: many_inputs(list(xs)), tracing_mode="real")(*big)
        op = next(iter(many_inputs._cache.values()))
        assert len(op._schema.arguments) == 200
        assert torch.allclose(gm(*big), sum(big[1:], big[0].clone()))
        assert len(call_nodes(gm, many_inputs)[0].args) == 200


if __name__ == "__main__":
    run_tests()
