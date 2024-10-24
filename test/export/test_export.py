# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import dataclasses
import io
import logging
import operator
import re
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from re import escape
from typing import Dict, List

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch._decomp import decomposition_table, get_decompositions
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import normalize_gm
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.utils import (
    _decomp_table_to_post_autograd_aten,
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._inductor.compile_fx import split_const_gm
from torch._subclasses import FakeTensorMode
from torch.export import default_decompositions, Dim, export, unflatten
from torch.export._trace import (
    _export,
    _export_to_torch_ir,
    DEFAULT_EXPORT_DYNAMO_CONFIG,
)
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM90OrLater,
)
from torch.testing._internal.common_device_type import onlyCPU, onlyCUDA
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    skipIfCrossRef,
    skipIfXpu,
    TEST_TRANSFORMERS,
    TestCase as TorchTestCase,
)
from torch.utils._pytree import (
    LeafSpec,
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
)


try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    HAS_TORCHREC = True
except ImportError:
    HAS_TORCHREC = False

try:
    from . import testing
except ImportError:
    import testing  # @manual=fbcode//caffe2/test:test_export-library
# The following import pattern matters as `test_export.export` is patched
# in other files (like test_export_nonstrict.py). `torch.export.export`
# will invalidate the patch.
from torch.export import export


torch.library.define("testlib::returns_tensor_symint", "(Tensor x) -> (Tensor, SymInt)")
torch.library.define(
    "testlib::foo",
    "(Tensor(a!) x, Tensor(b!) z) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_mutated",
    "(Tensor(a!) x) -> (Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_functional",
    "(Tensor x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_unbacked",
    "(Scalar x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("testlib::returns_tensor_symint", "cpu")
@torch.library.impl_abstract("testlib::returns_tensor_symint")
def returns_tensor_symint_impl(x):
    return x, x.shape[0]


@torch.library.impl("testlib::foo", "cpu")
@torch._dynamo.disable
def foo_impl(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z


@torch.library.impl_abstract("testlib::foo")
def foo_abstract(x, z):
    return x, z, x + z


@torch.library.impl("testlib::foo_mutated", "CompositeImplicitAutograd")
def foo_mutated(x):
    a, b, c = torch.ops.testlib.foo(x, x.cos())
    return a, a.cos()


@torch.library.impl("testlib::foo_functional", "CompositeImplicitAutograd")
def foo_functional(x):
    a, b, c = torch.ops.testlib.foo(x.cos(), x.cos())
    return a.cos()


@torch.library.impl("testlib::foo_unbacked", "CompositeImplicitAutograd")
def foo_unbacked(x):
    if x > 2:
        return torch.ones(4, 4)
    if x < 6:
        return torch.ones(4, 4)
    return torch.ones(4, 4)


@dataclass
class Inp:
    x: Tensor
    y: List[Tensor]
    z: Dict[str, Tensor]


NON_STRICT_SUFFIX = "_non_strict"
RETRACEABILITY_STRICT_SUFFIX = "_retraceability"
RETRACEABILITY_NON_STRICT_SUFFIX = "_retraceability_non_strict"
SERDES_SUFFIX = "_serdes"
PREDISPATCH_SUFFIX = "_pre_dispatch"
TRAINING_IR_DECOMP_STRICT_SUFFIX = "_training_ir_to_decomp"
TRAINING_IR_DECOMP_NON_STRICT_SUFFIX = "_training_ir_to_decomp_non_strict"


def is_non_strict_test(test_name):
    return test_name.endswith(NON_STRICT_SUFFIX)


def is_retracebility_test(test_name):
    return test_name.endswith(RETRACEABILITY_STRICT_SUFFIX) or test_name.endswith(
        RETRACEABILITY_NON_STRICT_SUFFIX
    )


def is_serdes_test(test_name):
    return test_name.endswith(SERDES_SUFFIX)


def is_training_ir_test(test_name):
    return test_name.endswith(TRAINING_IR_DECOMP_STRICT_SUFFIX) or test_name.endswith(
        TRAINING_IR_DECOMP_NON_STRICT_SUFFIX
    )


def get_hop_schema(ep: torch.export.ExportedProgram):
    hop_node = next(
        node
        for node in ep.graph.nodes
        if isinstance(node.target, torch._ops.HigherOrderOperator)
    )
    return torch._library.utils.hop_schema_from_fx_node(hop_node)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    def test_export_inline_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                torch._check_is_size(b)
                return torch.full((b, 1), 1)

        f = Module()
        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm.module()(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error_not_in_range(self):
        class InvalidInputConflictWithInputConstraints(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.zeros([3])
        dim_x = torch.export.Dim("dim_x", min=6)

        if is_non_strict_test(self._testMethodName):
            error_type = torch.fx.experimental.symbolic_shapes.ConstraintViolationError
        else:
            error_type = torch._dynamo.exc.UserError

        with self.assertRaisesRegex(error_type, "not in range"):
            export(
                InvalidInputConflictWithInputConstraints(),
                (inp,),
                dynamic_shapes={"x": {0: dim_x}},
            )

    def test_export_slice_maxsize(self):
        class Slice(torch.nn.Module):
            def forward(self, *args):
                return torch.ops.aten.slice.Tensor(*args)

        inp = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
        dynamic_shapes = (({0: Dim("dim")}, None, None, None),)
        torch.export.export(
            Slice(),
            inp,
            dynamic_shapes=dynamic_shapes,
        )

    def test_export_constraints_error(self):
        class ConflictingConstraints(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                torch._check_is_size(b)
                torch._check(b >= 4)
                torch._check(b <= 5)
                torch._check(b <= 5)
                torch._check(True)
                return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ep = export(ConflictingConstraints(), inp)

        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for expression u[\d+] \>\= 4"
        ):
            ep.module()(torch.tensor([3]))

    def test_export_assume_static_by_default(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                if x.shape[0] == 4:
                    return x + 1
                else:
                    return x

        branch_on_shape = Module()
        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):
    def _test_export_same_as_eager(self, f, args, kwargs=None):
        kwargs = kwargs or {}
        exported_program = export(f, args, kwargs)
        self.assertEqual(exported_program.module()(*args, **kwargs), f(*args, **kwargs))
        # this is not supported by .module()
        # reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        # self.assertEqual(
        #     exported_program.module()(*args, **reversed_kwargs), f(*args, **reversed_kwargs)
        # )

    def _check_dynamic_shapes_specs_and_shapes(
        self, model, inputs, specs, passing_shapes, failing_shapes, test_serdes=False
    ):
        from torch._export.serde.dynamic_shapes import (
            _dump_dynamic_shapes,
            _load_dynamic_shapes,
        )
        from torch.utils._pytree import tree_map

        def _construct_inputs(shapes):
            def _is_tensor_leaf(x):
                return isinstance(x, tuple) and all(isinstance(y, int) for y in x)

            return tree_map(
                lambda x: torch.randn(*x) if _is_tensor_leaf(x) else x,
                shapes,
                is_leaf=_is_tensor_leaf,
            )

        # exports with a list of equivalent dynamic shapes specs,
        # then tests for pass/fail on list of shapes
        for _specs in specs:
            ep = export(model, inputs, dynamic_shapes=_specs)
            eps = [ep]
            if test_serdes:
                # test dynamic shapes serialization
                # test that behavior remains the same when exporting with ser/des specs:
                # serialize + deserialize original specs, and export.
                ep_serdes = export(
                    model,
                    inputs,
                    dynamic_shapes=_load_dynamic_shapes(
                        _dump_dynamic_shapes(_specs, inputs)
                    ),
                )
                eps.append(ep_serdes)

            for ep in eps:
                for shapes in passing_shapes:
                    test_inputs = _construct_inputs(shapes)
                    ep.module()(*test_inputs)
                for shapes in failing_shapes:
                    test_inputs = _construct_inputs(shapes)
                    with self.assertRaises(RuntimeError):
                        ep.module()(*test_inputs)

    def test_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        self._test_export_same_as_eager(f, inp)

    @skipIfCrossRef
    def test_custom_tag_metadata_re_export(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 2))
                self.b = torch.nn.Parameter(torch.rand(4))

            def forward(self, x):
                out = torch.nn.functional.linear(x, self.w, self.b)
                return out

        f = Foo()
        inputs = (torch.zeros(1, 2),)
        ep = export(f, inputs)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        for node in new_gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"

        new_ep = ep._update(new_gm, ep.graph_signature)
        new_ep = export(new_ep.module(), inputs)
        self.assertEqual(new_ep.graph_module.meta["custom"]["f"], "bar")

        # the custom field should be preserved after re-export and
        # should not be copied to other nodes
        counter = 0
        for node in new_ep.graph.nodes:
            if "custom" in node.meta:
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
                self.assertTrue(node.target == torch.ops.aten.linear.default)

        self.assertEqual(counter, 1)

    def test_symint_output(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                z, y = x.size()
                return z + y + x[0], z

        inputs = (torch.ones(2, 3),)
        dim0_x, dim1_x = torch.export.dims("dim0_x", "dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        export(Foo(), inputs, dynamic_shapes=dynamic_shapes)

    def test_no_tensor_computation(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return y

        f = Module()
        inp = ([torch.ones(1, 3)], 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x_0 : [num_users=0] = placeholder[target=x_0]
    %y : [num_users=0] = placeholder[target=y]
    return (1,)""",
        )

    def test_no_tensor_computation_2(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x

        f = Module()
        inp = (torch.randn(3), 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=0] = placeholder[target=y]
    return (x,)""",
        )

    def test_no_tensor_computation_3(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return 5

        f = Module()
        inp = (2, 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=0] = placeholder[target=x]
    %y : [num_users=0] = placeholder[target=y]
    return (5,)""",
        )

    def test_no_tensor_computation_4(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x

        f = Module()
        inp = ([torch.randn(3)], 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x_0 : [num_users=1] = placeholder[target=x_0]
    %y : [num_users=0] = placeholder[target=y]
    return (x_0,)""",
        )

    def test_not_registered_parameter(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"foo": torch.nn.Parameter(torch.ones(3, 3))}

            def forward(self, x):
                return x + self.params["foo"]

        f = Basic()
        args = (torch.randn(1, 3),)
        # strict-mode will error out because foo is registered as parameter
        # in dynamo (a behavior that's different from eager). We decided to
        # follow eager behavior.
        ep = export(f, args, strict=False)
        gm = ep.module()
        self.assertEqual(len(ep.graph_signature.lifted_tensor_constants), 1)
        self.assertEqual(len(ep.graph_signature.parameters), 0)
        # check foo is not a parameter in the final graph
        self.assertEqual(len(list(gm.named_parameters())), 0)
        self.assertEqual(gm(*args), f(*args))
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %lifted_tensor_0 : [num_users=1] = get_attr[target=lifted_tensor_0]
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %lifted_tensor_0), kwargs = {})
    return (add,)""",
        )

    def test_external_call_non_strict_real_tensor(self):
        class ExternalMethod:
            def add(self, x):
                return x + x

        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.external_add = ExternalMethod().add

            def forward(self, x):
                return self.external_add(x)

        f = Basic()
        args = (torch.randn(1, 3),)
        ep = export(f, args, strict=False)
        self.assertEqual(ep.module()(*args), f(*args))

    def test_colon_parameter(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("foo:bar", torch.nn.Parameter(torch.ones(3, 3)))

            def forward(self, x):
                return x + getattr(self, "foo:bar")

        ep = export(M(), (torch.randn(3, 3),))
        x = torch.randn(3, 3)
        self.assertEqual(ep.module()(x), M()(x))

    def test_conv_dynamic(self):
        # Simple module for demonstration
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                a = self.conv(x)
                a.add_(y)
                return self.maxpool(self.relu(a))

        example_args = (torch.randn(2, 3, 256, 256), torch.ones(2, 32, 256, 256))
        dynamic_shapes = {"x": {0: Dim("batch")}, "y": {0: Dim("batch")}}
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))

        gm: torch.fx.GraphModule = torch.export.export_for_training(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        ).module()

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))

    def test_masked_select_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                mask = x.ge(0.5)
                return torch.masked_select(x, mask)

        example_args = (torch.randn(3, 4, 5),)
        dim0_x_max, dim1_x_max = 100, 7
        dynamic_shapes = {
            "x": {
                0: Dim("dim0_x", max=dim0_x_max),
                1: Dim("dim1_x_max", max=dim1_x_max),
            }
        }
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        # Test that the expected upper bound is among the range constraints.
        expected_upper_bound = dim0_x_max * dim1_x_max * 5
        vr_upper_bounds = [
            vr.upper for vr in exported_program.range_constraints.values()
        ]
        self.assertTrue(expected_upper_bound in set(vr_upper_bounds))
        # Test that none of the upper bounds are larger.
        for vr_upper in vr_upper_bounds:
            self.assertTrue(vr_upper <= expected_upper_bound)

    def test_setgrad_lifted_tensor(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                with torch.enable_grad():
                    c = torch.tensor(4)
                    z = c + x + y

                return z * z

        m = M()
        x = torch.randn(4)
        y = torch.randn(4)
        # Need to surround export with no_grad to bypass AutogradStateOpsFailSafeguard.
        with torch.no_grad():
            ep = export(m, (x, y))
        self.assertEqual(ep.module()(x, y), m(x, y))

    def test_basic_non_strict_real_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(1, 3))

            def forward(self, x, y):
                return x[0] + y - self.param

        f = Basic()
        args = ([torch.randn(1, 3)], torch.randn(1, 3))
        ep = export(f, args, strict=False)
        self.assertEqual(ep.module()(*args), f(*args))

    def test_basic_non_strict_fake_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x, y):
                return x[0] + y - self.param

        fake_mode = FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))
        f = Basic()
        with fake_mode:
            args = ([torch.empty(3, 2)], torch.empty(3, 2))
        ep = export(f, args, strict=False)
        inputs = ([torch.randn(3, 2)], torch.randn(3, 2))
        self.assertEqual(ep.module()(*inputs), f(*inputs))

    def test_non_strict_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.u = torch.nn.Buffer(torch.ones(1))
                self.v = torch.nn.Buffer(torch.ones(1))

            def forward(self, x, ys, zs, c):
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                self.v.add_(3)
                w = self.u - self.v
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y
                else:
                    return x - w, x - y

        foo = Foo()

        inp = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            [{0: dim}, {0: dim}],
            {"a": {0: dim}, "b": {0: dim}},
            None,
        )

        ep_ns = torch.export.export(
            foo, inp, dynamic_shapes=dynamic_shapes, strict=False
        )

        bad_runtime_inp1 = (
            torch.ones(6),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape(
                "Expected input at *args[1][0].shape[0] to be equal to 6, but got 5"
            ),
        ):
            ep_ns.module()(*bad_runtime_inp1)

        bad_runtime_inp2 = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(6),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[3].shape[0] to be equal to 4, but got 6"),
        ):
            ep_ns.module()(*bad_runtime_inp2)

        good_runtime_inp = (
            torch.ones(7),
            [torch.zeros(7), torch.ones(7)],
            {"a": torch.zeros(7), "b": torch.ones(7)},
            torch.ones(4),
        )
        ep_ns.module()(*good_runtime_inp)

        bad_example_inp = (
            torch.ones(2),
            [torch.zeros(2), torch.ones(2)],
            {"a": torch.zeros(2), "b": torch.ones(2)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            "2 not in range.*3,",
        ):
            ep_ns = torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_non_strict_dynamic_shapes_suggested_fixes(self):
        class Foo(torch.nn.Module):
            def forward(self, x, c):
                if x.shape[0] <= 6:
                    return x + 1, c + 2
                else:
                    return x - 1, c - 2

        foo = Foo()

        bad_example_inp = (
            torch.ones(5),
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            None,
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated \\(dim\\)!(.*\n)*.*"
            "Not all values of dim.*satisfy the generated guard(.*\n)*.*"
            "Suggested fixes:(.*\n)*.*"
            "dim = Dim\\('dim', min=3, max=6\\)",
        ):
            torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_unbacked_to_cond(self):
        class M(torch.nn.Module):
            def forward(self, a):
                az = a.nonzero()

                def true_fn(x):
                    return (x + 1).sum()

                def false_fn(x):
                    return (x + 3).sum()

                r = torch.cond(az.size(0) > 3, true_fn, false_fn, (az,))
                return r * 2

        M()(torch.randn(7))
        torch.export.export(M(), (torch.randn(7),))

    def test_unbacked_to_cond_passthrough(self):
        class M(torch.nn.Module):
            def forward(self, a):
                az = a.nonzero()

                def true_fn(x):
                    return x + 1

                def false_fn(x):
                    return x + 3

                r = torch.cond(az.size(0) > 3, true_fn, false_fn, (az,))
                return r * 2

        M()(torch.randn(7))
        torch.export.export(M(), (torch.randn(7),))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cond_contains_unbacked_no_escape(self):
        class M(torch.nn.Module):
            def forward(self, a, b1, b2, c):
                def true_fn(x):
                    return x * b1.item()

                def false_fn(x):
                    return x * b2.item()

                r = torch.cond(a, true_fn, false_fn, (c,))
                return r * 2

        args = (
            torch.tensor(True),
            torch.tensor([4]),
            torch.tensor([4]),
            torch.randn(10, requires_grad=True),
        )
        torch.export.export(M(), args)

    def test_cond_int_closure(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num = 4

            def forward(self, a, x):
                def true_fn(x):
                    return x * self.num

                def false_fn(x):
                    return x + self.num

                r = torch.cond(a, true_fn, false_fn, (x,))
                return r * 2

        args = (torch.tensor(True), torch.randn(10))
        ep = torch.export.export(M(), args)
        self.assertEqual(ep.module()(*args), M()(*args))

    def test_state_tensors(self):
        class M(torch.nn.Module):  # simple with register buffer
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(2, 3), persistent=False)

            def forward(self, x):
                # x = 2
                y = self.buf
                # y = 1
                w1 = self.buf + 3
                w2 = self.buf + 4
                w3 = self.buf + 5
                self.buf = w1
                z = self.buf
                self.buf = w3
                # z = 4
                return x + y + z + w2

        ep = export(M(), (torch.randn(2, 3),), strict=False)
        self.assertEqual(list(ep.graph_signature.buffers_to_mutate.values()), ["buf"])
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(2, 3) + 1), torch.ones(2, 3) * 12)
        )

        class M(torch.nn.Module):  # simple without register buffer
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.ones(2, 3)

            def forward(self, x):
                # x = 2
                y = self.buf
                # y = 1
                self.buf = self.buf + 3
                z = self.buf
                # z = 3
                return x + y + z

        with self.assertRaisesRegex(
            ValueError,
            "The tensor attribute self.buf was assigned during export",
        ):
            export(M(), (torch.randn(2, 3),), strict=False)

        class M(torch.nn.Module):  # complex with register buffer
            def __init__(self) -> None:
                super().__init__()
                tensors = [torch.ones(2, 3), torch.ones(2, 3)]
                for i, tensor in enumerate(tensors):
                    self.register_buffer(f"buf_{i}", tensor, persistent=False)

            def get_tensor(self, i):
                return getattr(self, f"buf_{i}")

            def set_tensor(self, i, val):
                setattr(self, f"buf_{i}", val)

            def forward(self, x):
                # x = 2
                y = self.get_tensor(0) + self.get_tensor(1)
                # y = 1 + 1
                self.set_tensor(0, torch.ones(2, 3) + 2)
                self.set_tensor(1, torch.ones(2, 3) + 2)
                z = self.get_tensor(0) + self.get_tensor(1)
                # z = 3 + 3
                return x + y + z

        ep = export(M(), (torch.randn(2, 3),), strict=False)
        self.assertEqual(
            list(ep.graph_signature.buffers_to_mutate.values()), ["buf_0", "buf_1"]
        )
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(2, 3) + 1), torch.ones(2, 3) * 10)
        )

        class M(torch.nn.Module):  # complex without register buffer
            def __init__(self) -> None:
                super().__init__()
                self.tensors = [torch.ones(2, 3), torch.ones(2, 3)]

            def get_tensor(self, i):
                return self.tensors[i]

            def set_tensor(self, i, val):
                self.tensors[i] = val

            def forward(self, x):
                # x = 2
                y = self.get_tensor(0) + self.get_tensor(1)
                # y = 1 + 1
                self.set_tensor(0, torch.ones(2, 3) + 2)
                self.set_tensor(1, torch.ones(2, 3) + 2)
                z = self.get_tensor(0) + self.get_tensor(1)
                # z = 3 + 3
                return x + y + z

        with self.assertRaisesRegex(
            ValueError,
            "The tensor attributes self.tensors\\[0\\], self.tensors\\[1\\] were assigned during export",
        ):
            export(M(), (torch.randn(2, 3),), strict=False)

    def test_state_primitives(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = 1
                self.y = {"k": 2}
                self.z = (3,)

            def forward(self, x):
                self.x = self.x + 4
                self.y["k"] = self.y["k"] + 5
                self.z = (self.z[0] + 6,)
                return x + self.x + self.y["k"] + self.z[0]

        ep = export(M(), (torch.randn(2, 3),))
        self.assertTrue(
            torch.allclose(ep.module()(torch.zeros(2, 3)), torch.ones(2, 3) * 21)
        )

    def test_state_shape_attribute_assignment(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.last_z_shape = self.linear.weight.shape

            def forward(self, x):
                self.last_z_shape = x.shape
                return self.linear(x)

        model = TestModule()
        x = torch.randn(20, 10)
        ep_model = export(model, (x,), strict=False).module()
        self.assertTrue(torch.allclose(model(x), ep_model(x)))

    def test_real_tensor_for_max_op(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x = x[x > 0]
                y = y[y > 0]
                return max(x.shape[0], y.shape[0])

        model = Foo()
        inputs = (torch.randn(64), torch.randn(64))
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

        self.assertEqual(ep.module()(*inputs), model(*inputs))
        x = torch.zeros(64)
        y = torch.ones(64)
        self.assertEqual(ep.module()(x, x), model(x, x))
        self.assertEqual(ep.module()(x, y), model(x, y))

    @testing.expectedFailureSerDer  # SymBool serialization? TODO(pianpwk)
    def test_real_tensor_bool_cast(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return bool(x.eq(0.1).any())

        model = Foo()
        inputs = (torch.randn(64),)
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs, strict=False)

    @testing.expectedFailureSerDer
    def test_is_nonzero(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.is_nonzero(x)

        def _long_tensor(nz):
            return torch.full((), int(nz))

        def _float_tensor(nz):
            return torch.full((), int(nz), dtype=torch.float32)

        def _bool_tensor(nz):
            return torch.full((), int(nz)).bool()

        mod = Foo()
        for _tensor in [
            _long_tensor,
            _float_tensor,
            _bool_tensor,
            # local_scalar_dense on complex NYI for fake tensors
        ]:
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                for nz in [True, False]:
                    sample_input = _tensor(nz=nz)
                    ep = export(mod, (sample_input,), strict=False)
                    self.assertEqual(ep.module()(sample_input), nz)
                    print(ep)

    def test_export_script_module(self):
        class Foo(torch.nn.Module):
            def forward(self, rv: torch.Tensor, t: torch.Tensor):
                i = t.item()
                return rv + i

        foo = Foo()
        foo_script = torch.jit.script(foo)
        inp = (torch.zeros(3, 4), torch.tensor(7))

        with self.assertRaisesRegex(
            ValueError, "Exporting a ScriptModule is not supported"
        ):
            export(foo_script, inp)

        from torch._export.converter import TS2EPConverter

        TS2EPConverter(foo_script, inp).convert()

    def test_dim_auto_and_dim(self):
        # test basic Dims
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        inputs = (torch.randn(4, 4), torch.randn(4, 4))
        shapes = {
            "x": (Dim.AUTO, Dim("d1", min=3)),
            "y": (Dim("d0", max=8), Dim.DYNAMIC),
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        x, y = [node for node in ep.graph.nodes if node.op == "placeholder"]
        self.assertEqual((s0 := x.meta["val"].shape[0]), y.meta["val"].shape[0])
        self.assertEqual((s1 := x.meta["val"].shape[1]), y.meta["val"].shape[1])
        vr0 = ep.range_constraints[s0.node.expr]
        vr1 = ep.range_constraints[s1.node.expr]
        self.assertEqual([vr0.upper, vr1.lower], [8, 3])

        # test derived Dims
        class Bar(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1::3] + z

        inputs = (torch.randn(4), torch.randn(13), torch.randn(4))
        dx = Dim("dx", min=2, max=10)
        shapes = {
            "x": (dx,),
            "y": (3 * dx + 1,),
            "z": (Dim.AUTO,),
        }
        ep = export(Bar(), inputs, dynamic_shapes=shapes)
        x, y, z = [node for node in ep.graph.nodes if node.op == "placeholder"]
        self.assertEqual((s0 := x.meta["val"].shape[0]), z.meta["val"].shape[0])
        expr = y.meta["val"].shape[0]
        free_symbols = expr.node.expr.free_symbols
        self.assertEqual(len(free_symbols), 1)
        self.assertEqual(next(iter(free_symbols)), s0.node.expr)

        # test specialization still complains
        inputs = (torch.randn(4), torch.randn(4))
        shapes = {
            "x": (Dim.STATIC,),
            "y": (Dim("dy"),),
        }
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Not all values of dy .* in the specified range are valid because dy was inferred to be a constant",
        ):
            export(Foo(), inputs, dynamic_shapes=shapes)

    def test_torch_fn(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.linear(x)
                x = self.relu(x)
                x = x + x
                return x

        ep1 = export(M1(), (torch.randn(3, 3),)).run_decompositions()
        expected_result = [
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_2", "builtin_function_or_method.linear"),
            ("linear_2", "builtin_function_or_method.linear"),
            ("relu_1", "function.relu"),
            ("add_1", "method_descriptor.add"),
        ]
        actual_result = []
        for i, node in enumerate(ep1.graph.nodes):
            if node.op == "call_function":
                actual_result.append(node.meta.get("torch_fn"))
        self.assertEqual(actual_result, expected_result)

        class M2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, weight, bias):
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.add(x, x)
                return x

        ep2 = export(
            M2(), (torch.randn(3, 3), torch.randn(3, 3), torch.randn(3))
        ).run_decompositions()
        expected_result = [
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_1", "builtin_function_or_method.linear"),
            ("relu_1", "function.relu"),
            ("add_1", "builtin_function_or_method.add"),
        ]
        actual_result = []
        for i, node in enumerate(ep2.graph.nodes):
            if node.op == "call_function":
                actual_result.append(node.meta.get("torch_fn"))
        self.assertEqual(actual_result, expected_result)

    @testing.expectedFailureSerDer  # failed serializing SymInt nodes in subgraph (known issue)
    def test_hoo_inline_users_issue(self):
        # This came from an issue where replace_with_hop passes would inline subgraphs,
        # and mess up node.users for nodes present in multiple subgraphs (e.g. _x in SetGradCase
        # below, since it's used in both set_grad_enabled HOO modules).
        # This checks that node.users and node.args are in correspondence.
        def check_users_for_graph(graph):
            def _tuple_contains(_tuple, val):
                # check nested, since output node args have format ((x, y, ...),)
                return any(
                    _tuple_contains(x, val) if isinstance(x, tuple) else x == val
                    for x in _tuple
                )

            for node in graph.nodes:
                # check node.users
                for user in node.users.keys():
                    assert _tuple_contains(user.args, node)
                # check node.args
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        assert _tuple_contains(arg.users, node)

        # check set grad enabled
        class SetGradCase(torch.nn.Module):
            def forward(self, x):
                _x = x.shape[0] + 2
                _xx = _x + 2
                with torch.no_grad():
                    y = _x * 4
                return _xx, y

        ep = export(
            SetGradCase(),
            (torch.randn(6),),
            dynamic_shapes={"x": (Dim("dx"),)},
            strict=False,
        )
        check_users_for_graph(ep.graph)

    @unittest.skipIf(IS_FBCODE, "Broken in fbcode")
    def test_export_predispatch_custom_ops_warnings(self):
        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        @foo.register_fake
        def _(x):
            return torch.empty_like(x)

        class Foo(torch.nn.Module):
            def forward(self, x):
                return foo(x)

        x = torch.randn(3)

        # Assert no warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch.export.export(Foo(), (x,))

        ops_registered_before = set(torch.ops.mylib)

        # Assert warning for CompositeImplictAutograd op
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            class Bar(torch.nn.Module):
                def forward(self, x):
                    return torch.ops.mylib.foo123(x)

            with self.assertWarnsRegex(
                UserWarning, "CompositeImplicitAutograd and have functional schema"
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    torch.export.export(Bar(), (x,))

        ops_registered_after = set(torch.ops.mylib)
        self.assertEqual(ops_registered_after, ops_registered_before)

    def test_export_preserve_linear_but_not_custom_op(self):
        table = torch.export.default_decompositions()
        del table[torch.ops.aten.linear.default]

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            class Bar(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(4, 4)

                def forward(self, x):
                    lin = self.linear(x)
                    return torch.ops.mylib.foo123(lin)

            x = torch.randn(4, 4)
            if IS_FBCODE:
                ep = export(Bar(), (x,)).run_decompositions(
                    decomp_table=None, _preserve_ops=(torch.ops.aten.linear.default,)
                )
            else:
                ep = export(Bar(), (x,)).run_decompositions(table)

            self.assertExpectedInline(
                str(ep.graph_module.code).strip(),
                """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    sin = torch.ops.aten.sin.default(linear);  linear = None
    return (sin,)""",
            )

    def test_export_preserve_linear_at_aot_level(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.ops.aten.chunk.default(x, 3, 0)

        ep = torch.export.export(Foo(), (torch.randn(3, 3),))
        if IS_FBCODE:
            ep = ep.run_decompositions(
                {}, _preserve_ops=(torch.ops.aten.linear.default,)
            )
        else:
            decomp_table = _decomp_table_to_post_autograd_aten()
            del decomp_table[torch.ops.aten.linear.default]
            ep = ep.run_decompositions(decomp_table)

        gm = ep.graph_module
        # linear is CompositeImplicitAutograd functional op so we should preserve it
        # chunk is CompositeImplicitAutograd non-functional op we decompose.
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    split = torch.ops.aten.split.Tensor(linear, 1);  linear = None
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    return (getitem, getitem_1, getitem_2)""",
        )

    def test_export_cond_preserve_torch_fn_for_subgraphs(self):
        class MySubModule(torch.nn.Module):
            def foo(self, x):
                return x.cos()

            def forward(self, x):
                return self.foo(x)

        class CondBranchClassMethod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.subm = MySubModule()

            def bar(self, x):
                return x.sin()

            def forward(self, x):
                return cond(x.sum() <= 2, self.subm.forward, self.bar, [x])

        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = CondBranchClassMethod()
        m.eval()
        gm = export(m, example_inputs).module()

        actual_torch_fns = []
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in {"sin", "cos"}:
                    torch_fn = node.meta.get("torch_fn")
                    print(torch_fn)
                    actual_torch_fns.append(torch_fn)
        exp_torch_fns = [
            ("cos_1", "method_descriptor.cos"),
            ("sin_1", "method_descriptor.sin"),
        ]
        self.assertEqual(actual_torch_fns, exp_torch_fns)

    def test_duplicate_modules_with_non_persistent_buffers(self):
        class FooWithBuf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.randn(4), persistent=False)

            def forward(self, x):
                return x + self.buf

        class BarWithFoo(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                return self.foo(x)

        class ModWith2Bars(torch.nn.Module):
            def __init__(self):
                super().__init__()
                foo = FooWithBuf()
                self.b1 = BarWithFoo(foo)
                self.b2 = BarWithFoo(foo)

            def forward(self, x):
                return self.b1(x) + self.b2(x)

        mod = ModWith2Bars()
        inputs = (torch.randn(4),)
        ep = export(mod, inputs)
        self.assertTrue(torch.allclose(ep.module()(*inputs), mod(*inputs)))

    def test_derived_dim_basic(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1:]

        foo = Foo()

        x, y = torch.randn(5), torch.randn(6)
        dimx = torch.export.Dim("dimx", min=3, max=6)

        dimy = torch.export.Dim("dimy", min=4, max=7)  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(dimy\\)!(.*\n)*.*"
                "The values of dimy.*must always be related to the values of dimx.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimy = dimx \\+ 1"
            ),
        ):
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        dimy = dimx * 2  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.* to be equal to 2\\*dimx, where dimx = 5, but got 6",
        ):
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        dimy = dimx + 1  # works
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 5, but got 6",
        ):
            ep.module()(torch.randn(4), torch.randn(6))

        self.assertEqual(ep.module()(torch.randn(4), torch.randn(5)).size()[0], 4)

    def test_derived_dim_nested(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1::2]

        foo = Foo()

        x, y = torch.randn(5), torch.randn(11)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimy = dimx * 2 + 1  # works
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        self.assertEqual(ep.module()(torch.randn(4), torch.randn(9)).size()[0], 4)

        class Foo(torch.nn.Module):
            def forward(self, z, y):
                return z[1:] + y[1::2]

        foo = Foo()

        z, y = torch.randn(6), torch.randn(11)

        dimz = dimx
        dimy = dimx * 2 - 1  # works
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

        dimz = dimx + 1
        dimy = dimx * 2 - 1  # doesn't work

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.*to be equal to 2\\*dimx - 1, where dimx = 5, but got 11",
        ):
            export(
                foo,
                (z, y),
                dynamic_shapes=({0: dimz}, {0: dimy}),
            )

        dimy = dimx * 2 + 1  # works
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        with self.assertRaisesRegex(
            RuntimeError, "Expected input.*shape.*to be <= 7, but got 8"
        ):
            ep.module()(torch.randn(8), torch.randn(15))
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 9, but got 8",
        ):
            ep.module()(torch.randn(5), torch.randn(8))

        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

    def test_derived_dim_integer(self):
        class Foo(torch.nn.Module):
            def forward(self, w):
                if w.shape[0] % 2 == 0:
                    return w[::2]
                else:
                    return w[1:-1:2]

        foo = Foo()

        w = torch.randn(10)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimw = dimx * 2 + 1  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected shape.*= 10 of input Tensor to be "
            "of the form 2\\*dimx \\+ 1, where dimx is an integer",
        ):
            export(
                foo,
                (w,),
                dynamic_shapes=({0: dimw},),
            )

        dimw = dimx * 2  # works
        ep = export(
            foo,
            (w,),
            dynamic_shapes=({0: dimw},),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*= 9 to be "
            "of the form 2\\*s1, where s1 is an integer",
        ):
            ep.module()(torch.randn(9))

        self.assertEqual(ep.module()(torch.randn(8)).size()[0], 4)
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be <= 12, but got 14",
        ):
            ep.module()(torch.randn(14))

    def test_derived_dim_repeat_derived(self):
        class Foo(torch.nn.Module):
            def forward(self, u, v):
                return u[::2] + v[::2]

        foo = Foo()

        u, v = torch.randn(10), torch.randn(10)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimw = dimx * 2  # works
        ep = export(
            foo,
            (u, v),
            dynamic_shapes=({0: dimw}, {0: dimw}),
        )
        self.assertEqual(ep.module()(torch.randn(8), torch.randn(8)).size()[0], 4)

    def test_derived_dim_out_of_order(self):
        dimy = torch.export.Dim("dimy", min=5, max=7)
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1:] + z[2:]

        foo = Foo()

        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 8, but got 5",
        ):
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )

    def test_derived_dim_out_of_order_repeat_derived(self):
        dimy = torch.export.Dim("dimy", min=5, max=7)
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2
        dimx1 = dimx
        dimx2 = dimz - 2  # works, effectively = dimx

        class Foo(torch.nn.Module):
            def forward(self, x, y, z, x1, x2):
                return x + y[1:] + z[2:] + x1 + x2

        foo = Foo()

        u, v, w, u1, u2 = (
            torch.randn(5),
            torch.randn(6),
            torch.randn(7),
            torch.randn(5),
            torch.randn(5),
        )
        ep = export(
            foo,
            (u, v, w, u1, u2),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 6, but got 5",
        ):
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

        ep = export(
            foo,
            (u, v, w, u, u),  # reused inputs
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 6, but got 5",
        ):
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

    def test_specialize_derived_dim_roots(self):
        # dim & derived dim both specialize
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x.reshape([-1]) + y

        dy = Dim("dy", min=6)
        x, y = torch.randn(6, 2), torch.randn(12)
        dynamic_shapes = {
            "x": (dy - 6, 2),
            "y": (dy,),
        }
        try:
            export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)
            raise Exception(
                "export() call should have failed with dynamic shapes error."
            )
        except torch._dynamo.exc.UserError as exc:
            expected_error_msg = (
                "Specializations unexpectedly required \(dy\)!(.*\n)*.*"
                ".*solving the guards generated for dy - 6.*resulted in a specialized value of 6(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                ".*dy = 12(.*\n)*.*"
            )
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            self.assertTrue(
                "dy - 6 = 6" not in exc.args[0]
            )  # don't suggest fix for non-root dim

    @unittest.skip("See https://github.com/pytorch/pytorch/issues/135759")
    def test_keep_composite_ops_invalid(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.ops.aten.chunk.default(x, 3, 0)

        def _(*args, **kwargs):
            return NotImplemented

        with self.assertWarnsRegex(UserWarning, "The op aten.chunk.default"):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({torch.ops.aten.chunk.default: _})

        with self.assertWarnsRegex(UserWarning, "The op aten.sym_size.default"):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({torch.ops.aten.sym_size.default: _})

        with self.assertWarnsRegex(
            UserWarning,
            "The op aten.native_batch_norm.default",
        ):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({torch.ops.aten.native_batch_norm.default: _})

    def test_keep_composite_ops_linear_convd(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = torch.export.export(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )
        if IS_FBCODE:
            ep_has_linear_convd = ep.run_decompositions(
                {},
                _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY,
            )
        else:
            ep_has_linear_convd = ep.run_decompositions({})

        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    linear = torch.ops.aten.linear.default(conv2d, c_linear_weight, c_linear_bias);  conv2d = c_linear_weight = c_linear_bias = None
    cos = torch.ops.aten.cos.default(linear);  linear = None
    sum_1 = torch.ops.aten.sum.default(conv1d);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        if IS_FBCODE:
            ep_has_convd = ep.run_decompositions(
                _preserve_ops=(
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.conv1d.default,
                )
            )
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.conv2d.default]
            del decomp_table[torch.ops.aten.conv1d.default]

            ep_has_convd = ep.run_decompositions(decomp_table=decomp_table)
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(conv1d, []);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )
        if IS_FBCODE:
            ep_has_convd = ep_has_convd.run_decompositions(
                _preserve_ops=(torch.ops.aten.conv2d.default,)
            )
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.conv2d.default]

            ep_has_convd = ep_has_convd.run_decompositions(decomp_table=decomp_table)
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    convolution = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution, []);  convolution = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

    def test_keep_composite_ops_linear_convd_for_training_ir(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Buffer(torch.randn(20, 98))
                self.bias = torch.nn.Buffer(torch.randn(20))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = torch.export.export_for_training(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )

        if IS_FBCODE:
            ep_has_linear_convd = ep.run_decompositions(
                {},
                _preserve_ops=testing._COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY,
            )
        else:
            ep_has_linear_convd = ep.run_decompositions(
                decomp_table={},
            )

        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    linear = torch.ops.aten.linear.default(conv2d, b_linear_weight, b_linear_bias);  conv2d = b_linear_weight = b_linear_bias = None
    cos = torch.ops.aten.cos.default(linear);  linear = None
    sum_1 = torch.ops.aten.sum.default(conv1d);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        if IS_FBCODE:
            ep_has_convd = ep.run_decompositions(
                _preserve_ops=(
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.conv1d.default,
                )
            )
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.conv2d.default]
            del decomp_table[torch.ops.aten.conv1d.default]

            ep_has_convd = ep.run_decompositions(decomp_table=decomp_table)

        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(b_linear_weight, [1, 0]);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, permute);  b_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(conv1d, []);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        if IS_FBCODE:
            ep_has_convd = ep_has_convd.run_decompositions(
                _preserve_ops=(torch.ops.aten.conv2d.default,)
            )
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.conv2d.default]
            ep_has_convd = ep_has_convd.run_decompositions(decomp_table=decomp_table)

        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    convolution = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(b_linear_weight, [1, 0]);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, permute);  b_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution, []);  convolution = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

    @unittest.skip("See https://github.com/pytorch/pytorch/issues/135759")
    def test_error_when_passing_mutating_primitive_op(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        ep = export(Foo(), (torch.ones(3, 3),))
        with self.assertWarnsRegex(
            UserWarning,
            "The op aten.index_put_.default",
        ):
            ep.run_decompositions({torch.ops.aten.index_put_.default: None})

    def test_export_cond_warns_constant_pred(self):
        class Mod(torch.nn.Module):
            def forward(self, pred, x):
                return torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))

        mod = Mod()
        with self.assertWarnsRegex(UserWarning, "Pred is a Python constant"):
            ep = export(mod, (True, torch.randn(3, 3)))

        nodes = ep.module().graph.find_nodes(
            op="call_function", target=torch.ops.aten.sin.default
        )
        self.assertEqual(len(nodes), 1)

    def test_export_custom_decomp_table_basic_pop(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            lib.define("foo456(Tensor x) -> Tensor")
            lib.impl("foo456", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            # Since this table hasn't been materialized yet, we shouldn't error
            val = table.pop(torch.ops.mylib.foo123.default)
            self.assertIsNotNone(val)

            with self.assertRaisesRegex(KeyError, "mylib.foo123.default"):
                table.pop(torch.ops.mylib.foo123.default)

            val = table.pop(torch.ops.mylib.foo123.default, "HELLO")
            self.assertEqual(val, "HELLO")

            all_ops = set(k for k, v in table.items())
            self.assertTrue(table.has_materialized)
            # When we force materialize, torch.ops.mylib.foo123.default should have gone
            self.assertFalse(torch.ops.mylib.foo123.default in all_ops)
            self.assertTrue(torch.ops.mylib.foo456.default in all_ops)

    def test_export_custom_decomp_table_container_methods(self):
        # tests __len__
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            table = default_decompositions()
            length_before = len(table)
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            lib.define("foo456(Tensor x) -> Tensor")
            lib.impl("foo456", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            self.assertEqual(len(table) - length_before, 2)

        # tests __contains__
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            self.assertTrue(torch.ops.mylib.foo123.default in table)
            del table[torch.ops.mylib.foo123.default]
            self.assertFalse(torch.ops.mylib.foo123.default in table)

        # Lot of ppl do
        # for op in all_ops:
        #     if op in table:
        #        del table[op]
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            if torch.ops.mylib.foo123.default in table:
                del table[torch.ops.mylib.foo123.default]

            self.assertFalse(torch.ops.mylib.foo123.default in table)
            table.materialize()
            self.assertFalse(torch.ops.mylib.foo123.default in table)

    def test_if_post_autograd_op_preserved(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.sum()

        ep = export(Foo(), (torch.ones(3, 3),))
        if IS_FBCODE:
            ep_preserve_sum = ep.run_decompositions(
                _preserve_ops=(torch.ops.aten.sum.default,)
            )
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.sum.default]
            ep_preserve_sum = ep.run_decompositions(decomp_table)

        # Even though we are decomposing to core aten which should make
        # sum into sum.dim_IntList, we explicitly marked it to not do that.
        self.assertExpectedInline(
            str(ep_preserve_sum.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x)
    sum_1 = torch.ops.aten.sum.default(x);  x = None
    add = torch.ops.aten.add.Tensor(sin, sum_1);  sin = sum_1 = None
    return (add,)""",
        )

        ep_no_preserve_sum = ep.run_decompositions()
        self.assertExpectedInline(
            str(ep_no_preserve_sum.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x)
    sum_1 = torch.ops.aten.sum.dim_IntList(x, []);  x = None
    add = torch.ops.aten.add.Tensor(sin, sum_1);  sin = sum_1 = None
    return (add,)""",
        )

    def test_set_grad_empty(self):
        class M(torch.nn.Module):
            def forward(self, x):
                with torch.no_grad():
                    x = x + 1
                    return x, None

        ep = export(M(), (torch.ones(3, 3),))
        inp = torch.randn(3, 3)
        self.assertTrue(torch.allclose(ep.module()(inp)[0], inp + 1))

    def test_derived_dim_out_of_order_simplified(self):
        _dimz = torch.export.Dim("_dimz", min=6, max=8)
        dimy = _dimz - 1
        dimx = dimy - 1
        dimz = torch.export.Dim("dimz", min=6, max=8)  # doesn't work, should be = _dimz

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1:] + z[2:]

        foo = Foo()
        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        try:
            export(
                foo,
                (u, v, w),
                dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
            )
        except torch._dynamo.exc.UserError as exc:
            expected_error_msg = (
                "Constraints violated \(dimz\)!(.*\n)*.*"
                "The values of dimz.*must always be related to the values of _dimz - 2.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimz = _dimz"
            )
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            # don't suggest fix for non-root dims, and no need to update root here
            self.assertTrue("_dimz - 2 = Dim(" not in exc.args[0])
            self.assertTrue("_dimz - 1 = _dimz - 1" not in exc.args[0])
            self.assertTrue("_dimz = Dim(" not in exc.args[0])

        dimz = dimx + 2  # works, effectively = _dimz
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 8, but got 5",
        ):
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )

    def test_simple_export_for_training(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        eager_model = Foo()
        ep_for_training = torch.export.export_for_training(
            eager_model, (torch.ones(2, 2),)
        )
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    return (linear,)""",
        )
        gm = ep_for_training.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    linear_weight = self.linear.weight
    linear_bias = self.linear.bias
    linear = torch.ops.aten.linear.default(x, linear_weight, linear_bias);  x = linear_weight = linear_bias = None
    return pytree.tree_unflatten((linear,), self._out_spec)""",
        )

        self.assertTrue(
            torch.allclose(gm(torch.ones(2, 2)), eager_model(torch.ones(2, 2)))
        )

    def test_export_for_training_with_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                x.add_(5)
                self.buffer.add_(5)
                return x + self.buffer

        eager_model_for_export = Foo()
        eager_model_for_testing = Foo()
        ep_for_training = torch.export.export_for_training(
            eager_model_for_export, (torch.ones(4, 4),)
        )
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, b_buffer, x):
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    add__1 = torch.ops.aten.add_.Tensor(b_buffer, 5);  b_buffer = None
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return (add,)""",
        )
        gm = ep_for_training.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    buffer = self.buffer
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    add__1 = torch.ops.aten.add_.Tensor(buffer, 5);  buffer = None
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

        self.assertTrue(
            torch.allclose(
                gm(torch.ones(4, 4)), eager_model_for_testing(torch.ones(4, 4))
            )
        )

    def test_export_for_training_with_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                x.add_(5)
                self.buffer.add_(5)
                return x + self.buffer.sum()

        eager_model_for_export_training = Foo()
        eager_model_for_export_inference = Foo()
        eager_model_for_testing = Foo()
        ep_for_training = torch.export.export_for_training(
            eager_model_for_export_training,
            (torch.ones(4, 4),),
            dynamic_shapes=({0: Dim("x")},),
        )

        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(torch.ones(2, 4)),
                eager_model_for_testing(torch.ones(2, 4)),
            )
        )

        ep_for_real = export(
            eager_model_for_export_inference,
            (torch.ones(4, 4),),
            dynamic_shapes=({0: Dim("x")},),
        )

        self.assertEqual(
            str(ep_for_training.range_constraints), str(ep_for_real.range_constraints)
        )

    def test_export_for_training_with_container_type(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, container):
                x = container[0][0]
                y = container[0][1]
                x.add_(5)
                y.add_(5)
                return x + y + self.buffer.sum()

        eager_model = Foo()
        ep_for_training = torch.export.export_for_training(
            eager_model,
            ([torch.ones(4, 4), torch.ones(4, 4)],),
        )

        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(
                    ([torch.ones(4, 4), torch.ones(4, 4)]),
                ),
                eager_model(([torch.ones(4, 4), torch.ones(4, 4)])),
            )
        )

    def test_export_for_training_run_decomp(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(2, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                self.buffer.add_(5)
                return self.linear(x) + self.buffer.sum()

        eager_model = Foo()
        ep_for_training = torch.export.export_for_training(
            eager_model,
            (torch.ones(2, 2),),
        )
        ep_for_inference = ep_for_training.run_decompositions()
        self.assertExpectedInline(
            str(ep_for_inference.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, b_buffer, x):
    add = torch.ops.aten.add.Tensor(b_buffer, 5);  b_buffer = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, x, permute);  p_linear_bias = x = permute = None
    sum_1 = torch.ops.aten.sum.dim_IntList(add, [])
    add_1 = torch.ops.aten.add.Tensor(addmm, sum_1);  addmm = sum_1 = None
    return (add, add_1)""",
        )

    def test_derived_dim_out_of_order_simplified_repeat_non_derived(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, y1, z):
                return x + y[1:] + y1[1:] + z[2:]

        foo = Foo()

        u, v, v1, w = torch.randn(5), torch.randn(6), torch.randn(6), torch.randn(7)
        _dimz = torch.export.Dim("_dimz", min=6, max=8)
        dimy = _dimz - 1
        dimx = dimy - 1
        dimz = dimx + 2  # works, effectively = _dimz
        ep = export(
            foo,
            (u, v, v1, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected input.*shape.*to be equal to 7, but got 5",
        ):
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(5),
                torch.randn(8),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(7),
                torch.randn(8),
            ).size()[0],
            6,
        )

    def test_static_dim_constraints(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(6, 4)

            def forward(self, x, y, z):
                x0 = self.l(x) + y[1:]
                return x0, z * 2.0

        foo = Foo()
        inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))
        dx = Dim("dx", min=3, max=6)
        dy = dx + 1
        dz = Dim("dz", min=3, max=6)

        # test that tweaking shapes fails
        wrong_shape_inputs = [
            (torch.randn(4, 7), torch.randn(5, 4), torch.randn(3, 3)),
            (torch.randn(4, 6), torch.randn(5, 5), torch.randn(3, 3)),
            (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 4)),
        ]

        # all of these should be fine
        for dynamic_shapes in [
            ({0: dx, 1: 6}, {0: dy, 1: 4}, {0: dz, 1: 3}),
            ((dx, None), (dy, 4), (dz, 3)),
            ((None, 6), (5, None), (None, None)),
            ((4, 6), {0: None, 1: 4}, {0: None, 1: 3}),
            (None, None, (Dim.STATIC, Dim.STATIC)),
        ]:
            ep = export(foo, inputs, dynamic_shapes=dynamic_shapes)
            self.assertEqual(foo(*inputs), ep.module()(*inputs))
            for wrong_inputs in wrong_shape_inputs:
                with self.assertRaises(RuntimeError):
                    ep.module()(*wrong_inputs)

        # check range_constraints - static dims shouldn't be present
        ep = export(foo, inputs, dynamic_shapes=((dx, None), (dy, 4), (dz, 3)))
        self.assertEqual(len(ep.range_constraints), 3)
        for vr in ep.range_constraints.values():
            self.assertTrue(vr.lower < vr.upper)

        # check raised errors
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 5 does not match input size of 4, for .*",
        ):
            _ = export(foo, inputs, dynamic_shapes=((5, None), None, None))
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 9 does not match input size of 6, for .*",
        ):
            _ = export(foo, inputs, dynamic_shapes=((dx, 9), (dy, 4), (3, 3)))

    def test_dim_1_2(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * 2

        dx = Dim("dx", min=1, max=2)
        ep = export(Foo(), (torch.randn(2, 2),), dynamic_shapes=({0: dx, 1: None},))
        ep.module()(torch.randn(1, 2))
        ep.module()(torch.randn(2, 2))
        with self.assertRaisesRegex(
            RuntimeError, "Expected input at .* to be <= 2, but got 3"
        ):
            ep.module()(torch.randn(3, 2))
        vr = list(ep.range_constraints.values())[0]
        self.assertEqual(vr.lower, 1)
        self.assertEqual(vr.upper, 2)

    def test_derived_dim_1_2(self):
        class Bar(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1:]

        dx = Dim("dx", min=1, max=2)
        ep = export(
            Bar(),
            (torch.randn(2, 2), torch.randn(3, 2)),
            dynamic_shapes=({0: dx, 1: None}, {0: dx + 1, 1: None}),
        )
        ep.module()(torch.randn(1, 2), torch.randn(2, 2))
        range_lower_bounds = sorted(vr.lower for vr in ep.range_constraints.values())
        range_upper_bounds = sorted(vr.upper for vr in ep.range_constraints.values())
        self.assertEqual(range_lower_bounds, [1, 2])
        self.assertEqual(range_upper_bounds, [2, 3])

    def test_dynamic_shapes_builder_basic(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[0] + z["k"]

        m = M()

        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x, y, z)

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    @testing.expectedFailureRetraceabilityNonStrict
    def test_dynamic_shapes_builder_kwargs(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[0] + z["k"]

        m = M()

        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x,)
        kwargs = {"z": z, "y": y}

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, kwargs=kwargs, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    # retracing doesn't seem to like dataclass registration,
    # raising a dynamo error in fx_pytree.tree_flatten_spec
    @testing.expectedFailureRetraceability
    def test_dynamic_shapes_builder_pytree(self):
        torch.export.register_dataclass(
            Inp,
            serialized_type_name="test_dynamic_shapes_builder_pytree.Inp",
        )

        class M(torch.nn.Module):
            def forward(self, inp: Inp):
                return inp.x + inp.y[0] + inp.z["k"]

        m = M()
        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (Inp(x, y, z),)

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, dynamic_shapes=shapes_collection.dynamic_shapes(m, args))
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    def test_mismatched_dynamic_shapes(self):
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        class M(torch.nn.Module):
            def forward(self, x):
                return x["k"]["k"][0] + x["k"]["k"][1]

        inputs = ({"k": {"k": [torch.rand(4), torch.rand(4)]}},)
        dim = torch.export.Dim("dim")

        dynamic_shapes = {
            "k": {"k": [dim, dim]}
        }  # ValueError: Node keys mismatch; missing key(s): {'x'}; extra key(s): {'k'}.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "When `dynamic_shapes` is specified as a dict, its top-level keys "
                "must be the arg names ['x'] of `inputs`, but here they are ['k']. "
                "Since here `inputs` is a list/tuple enclosing a single dict, "
                "maybe you just forgot to enclose `dynamic_shapes` in a list/tuple?"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": [dim, dim]}},
        )  # torch._dynamo.exc.UserError: Unexpected dynamic_shape .*dim.* of Tensor, try None instead
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Unexpected input tensor shape .*dim.* "
            + re.escape(
                "specified at `dynamic_shapes[0]['k']['k'][0]` "
                "(expected either a list/tuple of dimensions, or a dict mapping indices to dimensions,"
                " where each dimension is an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC)"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": (dim, dim)}},
        )  # ValueError: Node type mismatch; expected <class 'list'>, but got <class 'tuple'>.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']['k']` is a <class 'list'>, but `dynamic_shapes[0]['k']['k']` is a <class 'tuple'>"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = ({"k": {"k": [(dim,), (dim,)]}},)  # ok
        export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": dim}},
        )  # ValueError: Node type mismatch; expected <class 'list'>, but got .*_Dim.*.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']['k']` is a <class 'list'>, but `dynamic_shapes[0]['k']['k']` is not"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = {
            "x": {"k": [(dim,), (dim,)]},
            "k": {"k": [(dim,), (dim,)]},
        }  # ValueError: Node arity mismatch; expected 1, but got 2.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "When `dynamic_shapes` is specified as a dict, its top-level keys "
                "must be the arg names ['x'] of `inputs`, but here they are ['x', 'k']. "
                "Alternatively, you could also ignore arg names entirely "
                "and specify `dynamic_shapes` as a list/tuple matching `inputs`."
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": [(dim,), (dim,), (dim,)]}},
        )  # ValueError: Node arity mismatch; expected 2, but got 3.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']['k']` has 2 elements, but `dynamic_shapes[0]['k']['k']` has 3 elements"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"K": [(dim,), (dim,), (dim,)]}},
        )  # ValueError: Node keys mismatch; missing key(s): {'k'}; extra key(s): {'K'}.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']` has keys ['k'], but `dynamic_shapes[0]['k']` has keys ['K']"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        class N(torch.nn.Module):
            def forward(self, x):
                return x["k"]["k1"][0] + x["k"]["k2"][0]

        inputs = ({"k": {"k1": [torch.rand(4)], "k2": [torch.rand(4)]}},)
        dim = torch.export.Dim("dim")

        dynamic_shapes = ({"k": {"k2": [(dim,)], "k1": [(dim,)]}},)  # ok
        export(N(), inputs, dynamic_shapes=dynamic_shapes)

    @testing.expectedFailureSerDer  # no unbacked bindings after deserialization?
    def test_unbacked_bindings_for_divisible_u_symint(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            class M(torch.nn.Module):
                def forward(self, a, b):
                    return torch.ops.mylib.foo(a, b)

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a, b):
                return a[b.item()]

            @torch.library.register_fake("mylib::foo", lib=lib)
            def foo_fake_impl(a, b):
                ctx = torch.library.get_ctx()
                u = ctx.new_dynamic_size(min=0, max=len(a) // 10) * 10
                return torch.empty(u, a.shape[1], dtype=a.dtype)

            ep = export(
                M(),
                (torch.randn(100, 4), torch.tensor(10)),
            )
            foo = [node for node in ep.graph.nodes if node.name == "foo"][0]
            unbacked_bindings = foo.meta["unbacked_bindings"]
            self.assertEqual(len(unbacked_bindings), 1)  # check binding is {u: path}
            u = next(iter(unbacked_bindings.keys()))
            self.assertEqual(
                type(u).__name__, "Symbol"
            )  # check binding is symbol, not expr
            path = unbacked_bindings[u]
            self.assertEqual(len(path), 3)  # check path is [size, 0, DivideByKey(10)]
            self.assertEqual(type(path[2]).__name__, "DivideByKey")
            self.assertEqual(path[2].divisor, 10)

    def test_torch_check_eq_commutativity(self):
        class M1(torch.nn.Module):
            def forward(self, x1, x2, x3, y):
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # instead of: torch._check((z2 + z3) == z1)
                torch._check(z1 == (z2 + z3))
                if z2 + z3 == z1:
                    return y * 2
                else:
                    return y + 3

        export(
            M1(),
            (torch.tensor(6), torch.tensor(3), torch.tensor(3), torch.randn(1)),
        )

        class M2(torch.nn.Module):
            def forward(self, x1, x2, x3, y):
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # instead of: torch._check((z2 + z3) != z1)
                torch._check(z1 != (z2 + z3))
                if z2 + z3 == z1:
                    return y * 2
                else:
                    return y + 3

        export(
            M2(),
            (torch.tensor(6), torch.tensor(6), torch.tensor(6), torch.randn(1)),
        )

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        with self.assertRaisesRegex(
            (
                torchdynamo.exc.UserError,
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
            ),
            "Could not guard on data-dependent expression",
        ):
            _ = export(M(), (torch.tensor([2, 3, 5]),))

    def test_suggested_fixes_for_data_dependent_errors_basic(self):
        # suggested fixes for data-dependent errors only work in non-strict mode
        strict = False
        error_type = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode

        # Just to introduce some indirection: N is a top-level module N that calls
        # module M, defined next.
        class N(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = M()

            def forward(self, t):
                return self.m(t) + 1

        # example input
        t = torch.tensor([1, 4, 4], dtype=torch.int32)

        # We define a series of versions of M() below. Each version has
        # raises a data-dependent error that the next version fixes, by
        # copy-pasting a suggested fix in the error message. The fix is
        # always a torch.check() on an unresolved condition (or its negation)
        # on unbacked symints mentioned in the error message.
        # Note that the suggested fixes are in terms of local variables
        # near the location of error that "contain" the unbacked symints
        # in the unresolved condition (either directly or indirectly, e.g.,
        # inside a list or inside the shape of a tensor).

        class M_v0(torch.nn.Module):
            def forward(self, t):
                items = [t[i].item() for i in range(t.numel())]
                r = torch.randn([items[0], items[1]])
                # Could not guard on data-dependent expression Eq(u2, -1)
                return r.view(items[0], items[2])

        M = M_v0
        with self.assertRaisesRegex(
            error_type,
            "The following call raised this error(.*\n)+"
            f".*{re.escape('return r.view(items[0], items[2])')}(.*\n)+"
            "To fix the error, insert one of the following checks before this call.*:\n"
            f".*{re.escape('torch._check(items[2] == (-1))')}.*\n"
            f".*{re.escape('torch._check(items[2] != (-1))')}(.*\n)+"
            f".*{re.escape('(These suggested fixes were derived by replacing `u2` with items[2] in Eq(u2, -1) and its negation.)')}",
        ):
            export(N(), (t,), strict=strict)

        class M_v1(torch.nn.Module):
            def forward(self, t):
                items = [t[i].item() for i in range(t.numel())]
                r = torch.randn([items[0], items[1]])
                # Could not guard on data-dependent expression Eq(u2, -1)
                torch._check(items[2] != -1)
                # Could not guard on data-dependent expression u2 >= 0
                return r.view(items[0], items[2])

        M = M_v1
        with self.assertRaisesRegex(
            error_type,
            "The following call raised this error(.*\n)+"
            f".*{re.escape('return r.view(items[0], items[2])')}(.*\n)+"
            "To fix the error, insert one of the following checks before this call.*:\n"
            f".*{re.escape('torch._check(items[2] >= 0)')}.*\n"
            f".*{re.escape('torch._check(items[2] < 0)')}(.*\n)+"
            f".*{re.escape('(These suggested fixes were derived by replacing `u2` with items[2] in u2 >= 0 and its negation.)')}",
        ):
            export(N(), (t,), strict=strict)

        class M_v2(torch.nn.Module):
            def forward(self, t):
                items = [t[i].item() for i in range(t.numel())]
                r = torch.randn([items[0], items[1]])
                # Could not guard on data-dependent expression Eq(u2, -1)
                torch._check(items[2] != -1)
                # Could not guard on data-dependent expression u2 >= 0
                torch._check(items[2] >= 0)
                # Could not guard on data-dependent expression Eq(u1, u2)
                return r.view(items[0], items[2])

        M = M_v2
        with self.assertRaisesRegex(
            error_type,
            "The following call raised this error(.*\n)+"
            f".*{re.escape('return r.view(items[0], items[2])')}(.*\n)+"
            "To fix the error, insert one of the following checks before this call.*:\n"
            f".*{re.escape('torch._check(items[2] == items[1])')}.*\n"
            f".*{re.escape('torch._check(items[2] != items[1])')}(.*\n)+"
            f".*{re.escape('(These suggested fixes were derived by replacing `u1` with items[1] or r.shape[1], `u2` with items[2] in Eq(u2, u1) and its negation.)')}",
        ):
            export(N(), (t,), strict=strict)

        class M_v3(torch.nn.Module):
            def forward(self, t):
                items = [t[i].item() for i in range(t.numel())]
                r = torch.randn([items[0], items[1]])
                # Could not guard on data-dependent expression Eq(u2, -1)
                torch._check(items[2] != -1)
                # Could not guard on data-dependent expression u2 >= 0
                torch._check(items[2] >= 0)
                # Could not guard on data-dependent expression Eq(u1, u2)
                torch._check(items[2] == r.shape[1])
                return r.view(items[0], items[2])

        M = M_v3
        export(N(), (t,), strict=strict)

    @testing.expectedFailureSerDer  # T195866111
    def test_suggested_fixes_for_data_dependent_errors_puzzlers(self):
        # suggested fixes for data-dependent errors only work in non-strict mode
        strict = False
        error_type = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode

        def retry_export(m, inp, fixes):
            # API that applies a series of fixes, retrying export after applying each fix,
            # and asserting the applied fix was suggested in the previous try.
            # Using this API avoids the need to define multiple versions of the same test
            # module, as in `test_suggested_fixes_for_data_dependent_errors_basic` above.
            def code(snippets):
                return f"[{', '.join(snippets)}]"

            for i in range(len(fixes)):
                with self.assertRaisesRegex(error_type, re.escape(fixes[i])):
                    export(m, (*inp, code(fixes[:i])), strict=strict)
            export(m, (*inp, code(fixes)), strict=strict)

        # The following examples are lifted from @ezyang's "Data-dependent shape puzzlers"
        # notebook at https://www.internalfb.com/intern/anp/view/?id=5330476

        # These test modules are written in a way that works well with retry_export above.
        # Specifically, they take an extra `fixes` argument and `eval` it at the location
        # that is expected to raise errors.

        class cf_implicitsize(torch.nn.Module):
            def forward(self, x, y, fixes):
                i = x.item()
                eval(fixes)
                # instead of y[i]
                return y.narrow(0, i, 1).squeeze()

        retry_export(
            cf_implicitsize(),
            (torch.tensor(2), torch.randn(10)),
            fixes=[
                # Could not guard on data-dependent expression u0 < 0
                "torch._check(i >= 0)",
            ],
        )

        class cf_nomemo(torch.nn.Module):
            def forward(self, x, y, fixes):
                i = y[0].item()
                eval(fixes)
                return x.unsqueeze(1).expand(-1, i)

        retry_export(
            cf_nomemo(),
            (torch.randn(8), torch.tensor([2])),
            fixes=[
                # Could not guard on data-dependent expression Eq(u0, 1)
                "torch._check(i != 1)",
                # Could not guard on data-dependent expression Ne(u0, -1)
                "torch._check(i != (-1))",
            ],
        )

        class cf_changevar(torch.nn.Module):
            def forward(self, x, fixes):
                i = x.item()
                eval(fixes)
                r = torch.arange(i // 2)
                return r + r

        retry_export(
            cf_changevar(),
            (torch.tensor(20),),
            fixes=[
                # Could not guard on data-dependent expression Eq((u0//2), 0)
                "torch._check(((i//2)) != 0)",
                # Could not guard on data-dependent expression Eq((u0//2), 1)
                "torch._check(((i//2)) != 1)",
            ],
        )

        class cf_stacklist(torch.nn.Module):
            def forward(self, xs, y, fixes):
                i = y.item()
                eval(fixes)
                # instead of xs[i]
                return torch.stack(xs, 0).narrow(0, i, 1).squeeze()

        retry_export(
            cf_stacklist(),
            ([torch.ones(5) * i for i in range(10)], torch.tensor(2)),
            fixes=[
                # Could not guard on data-dependent expression u0 < 0
                "torch._check(i >= 0)",
            ],
        )

        class cf_tensorsplit(torch.nn.Module):
            def forward(self, x, offsets_t, fixes):
                lengths = torch.diff(offsets_t).tolist()
                rs = []
                start = 0
                for length in lengths:
                    eval(fixes)
                    rs.append(x.narrow(0, start, length))
                    start += length
                return rs

        retry_export(
            cf_tensorsplit(),
            (torch.arange(10), torch.tensor([0, 2, 5, 7, 10])),
            fixes=[],  # nothing to fix!
        )

    def test_no_suggested_fixes_for_data_dependent_errors(self):
        # suggested fixes for data-dependent errors only work in non-strict mode
        strict = False
        error_type = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode

        class cf_stacklist(torch.nn.Module):
            def forward(self, xs, y):
                # y.item() is not a local, so we can't suggest a fix
                return torch.stack(xs, 0).narrow(0, y.item(), 1).squeeze()

        with self.assertRaisesRegex(
            error_type,
            "Could not guard on data-dependent expression u0 < 0",
        ):
            export(
                cf_stacklist(),
                ([torch.ones(5) * i for i in range(10)], torch.tensor(2)),
                strict=strict,
            )

        class Box:
            def __init__(self, content):
                self.content = content

        from torch.utils._pytree import register_pytree_node

        register_pytree_node(
            Box,
            lambda box: ([box.content], None),  # flatten_fn
            lambda contents, _context: Box(*contents),  # unflatten_fn
            flatten_with_keys_fn=None,  # unflatten_fn
            serialized_type_name="test_no_suggested_fixes_for_data_dependent_errors.Box",
        )

        class cf_stacklist_udd(torch.nn.Module):
            def forward(self, xs, y):
                box = Box(y.item())
                # box.content is not a local, so we can't suggest a fix
                return torch.stack(xs, 0).narrow(0, box.content, 1).squeeze()

        with self.assertRaisesRegex(
            error_type,
            "Could not guard on data-dependent expression u0 < 0",
        ):
            export(
                cf_stacklist_udd(),
                ([torch.ones(5) * i for i in range(10)], torch.tensor(2)),
                strict=strict,
            )

    def test_tolist(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.tolist()

        ep = export(M(), (torch.ones(3, dtype=torch.int),))
        self.assertEqual(ep.module()(torch.tensor([1, 2, 3])), [1, 2, 3])

    def test_if_functional(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                z = x + 4
                z.add_(4)
                y = z.view(x.shape)
                return x.cos() + y.cos()

        foo = Module()
        gm = export(foo, (torch.tensor([2, 3, 5]),))

        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph.",
                )
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.view.default
            ):
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)

    def test_solver_unsupported_sympy_function(self):
        # repro of https://github.com/pytorch/pytorch/issues/131897

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = torch.nn.functional.interpolate(
                    x, scale_factor=0.5, mode="bilinear"
                )
                x = torch.nn.functional.interpolate(
                    x, scale_factor=2.0, mode="bilinear"
                )
                x = x + y
                return x

        model = MyModule().eval()

        inputs = (
            torch.rand((1, 1, 32, 32)),
            torch.rand((1, 1, 32, 32)),
        )

        dim = torch.export.Dim("Dim", min=16, max=64)
        dynamic_shapes = {"x": {2: dim, 3: dim}, "y": {2: dim, 3: dim}}

        exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)
        self.assertEqual(exported_program.module()(*inputs), model(*inputs))

    def test_export_mod_constraints(self):
        class BasicDynamiShapeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)

        m = BasicDynamiShapeModel()
        a = torch.randn(3, 4)
        dim0_x = torch.export.Dim("dim0_x", min=3)
        dim1_x = torch.export.Dim("dim1_x", max=8000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        em = torch.export._trace._export(
            m,
            (a,),
            dynamic_shapes=dynamic_shapes,
            allow_complex_guards_as_runtime_asserts=True,
        )
        em.module()(torch.randn(4, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(Mod\(s0\*s1, s0 \- 1\), 0\)",
        ):
            em.module()(torch.randn(4, 5))

        dim0_x = None
        dim1_x = 2 * torch.export.Dim("_dim1_x", max=4000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        em = torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)
        x = torch.randn(3, 5)
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected.*shape\\[1\\] = 5 to be of the form 2\\*s1, where s1 is an integer",
        ):
            em.module()(x)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_dont_duck_size_for_auto_dynamic(self):
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                # x: [s0, s1], y: [s0 + 1, 4]
                assert y.shape[1] == 4
                assert x.shape[0] == y.shape[0] - 1
                return x * 2, y * 2

        # duck sizing would make all static based on these sample inputs
        inputs = (torch.randn(4, 4), torch.randn(5, 4))
        shapes = {
            "x": (AUTO, AUTO),
            "y": (AUTO, AUTO),
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        ep.module()(torch.randn(6, 3), torch.randn(7, 4))

    @testing.expectedFailureRetraceability  # T183144629
    def test_map(self):
        class Module(torch.nn.Module):
            def forward(self, xs, y, z):
                def body(x, y, z):
                    return x + y + z

                return map(body, xs, y, z)

        list_tensor_map = Module()
        inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
        self._test_export_same_as_eager(list_tensor_map, inps)

    @unittest.expectedFailure
    def test_crop_like(self):
        # https://fb.workplace.com/groups/1405155842844877/posts/8195050017188725/

        # Minimal crop code copied from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional
        class CropLike(torch.nn.Module):
            def forward(self, image, crop_height, crop_width):
                c, image_height, image_width = image.shape
                crop_top = int(round((image_height - crop_height) / 2.0))
                crop_left = int(round((image_width - crop_width) / 2.0))
                return image[
                    ...,
                    crop_top : crop_top + crop_height,
                    crop_left : crop_left + crop_width,
                ]

        crop = CropLike()
        imagew = Dim("width")
        imageh = Dim("height")
        dynamic_dims = {
            "image": {0: None, 1: imageh, 2: imagew},
            "crop_height": None,
            "crop_width": None,
        }
        args = (torch.rand(3, 512, 512), 150, 150)
        ecrop = export(crop, args=args, dynamic_shapes=dynamic_dims)

        args = (torch.rand(3, 700, 700), 150, 150)
        self.assertEqual(ecrop.module()(*args), ecrop(*args))

    @testing.expectedFailureRetraceabilityNonStrict
    def test_export_func_with_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, kw1, kw2):
                return arg1 + arg2, kw1 + kw2

        kw_func = Module()
        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_export_func_with_pytree_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b):
                return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "a": {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)},
            "b": [torch.ones(2, 3), torch.ones(3, 4)],
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_export_func_with_default_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b=1):
                return arg1 + arg2, a["kw1"] + a["kw2"] + b

        kw_func = Module()

        class Module2(torch.nn.Module):
            def forward(self, arg1, arg2, a=1, b=2):
                return arg1 + a, arg2 + b

        kw_func2 = Module2()

        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs1 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)
        self._test_export_same_as_eager(kw_func, args, kwargs2)
        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)

    def test_export_func_with_var_postional_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args):
                return arg1 + args[0], arg2 + args[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_export_func_with_keyword_only_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2):
                return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_export_func_with_var_keyword_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                return (
                    arg1 + args[0] + kw1 + kwargs["kw3"],
                    arg2 + args[1] + kw2 + kwargs["kw4"],
                )

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "kw1": torch.ones(2, 3),
            "kw2": torch.ones(3, 4),
            "kw3": torch.ones(2, 3),
            "kw4": torch.ones(3, 4),
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_unbacked_slice(self):
        class M(torch.nn.Module):
            def forward(self, scores, score_thr, topk: torch.Tensor, results=None):
                valid_mask = scores > score_thr
                scores = scores[valid_mask]
                valid_idxs = torch.nonzero(valid_mask).to(scores.device)

                num_topk = torch.minimum(topk, torch.tensor(valid_idxs.shape[0])).item()
                torch._check_is_size(num_topk)
                torch._check(scores.shape[0] >= num_topk)
                scores, idxs = scores.sort(descending=True)
                scores = scores[:num_topk]
                topk_idxs = valid_idxs[idxs[:num_topk]]
                keep_idxs, labels = topk_idxs.unbind(dim=1)

                return scores, labels, keep_idxs

        score = torch.tensor(
            [[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08], [0.4, 0.1, 0.08]]
        )
        bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
        score_thr = 0.15
        nms_pre = torch.tensor(4)
        inputs = (score, score_thr, nms_pre, dict(bbox_pred=bbox_pred))

        ep = torch.export.export(M(), inputs)
        orig_res = M()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    def test_unflatten_asserts(self):
        # TODO: strict-export fails
        class M1(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()

                torch._check_is_size(b)
                torch._check(b < y.size(0))
                return y[:b]

        class M3(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()

                torch._check_is_size(b)
                torch._check(b < y.size(0) * 2)
                return y[:b]

        class M2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M1()
                self.m3 = M3()

            def forward(self, x, y):
                return self.m1(x, y) + self.m3(x, y)

        inputs = (torch.tensor(3), torch.randn(10))

        ep = torch.export.export(
            M2(), inputs, dynamic_shapes={"x": None, "y": (Dim("moo"),)}, strict=False
        )
        orig_res = M2()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

        unflattened = torch.export.unflatten(ep)
        ep_res = unflattened(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    @testing.expectedFailureRetraceabilityNonStrict
    def test_export_func_with_var_keyword_pytree_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                return (
                    arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0],
                    arg2[1] + args[1] + kw2 + kwargs["kw4"],
                )

        kw_func = Module()
        args = (
            torch.ones(2, 3),
            [(torch.ones(2, 3),), torch.ones(3, 4)],
            torch.ones(2, 3),
            torch.ones(3, 4),
        )
        kwargs = {
            "kw1": (torch.ones(2, 3),),
            "kw2": torch.ones(3, 4),
            "kw3": (torch.ones(2, 3), torch.ones(3, 4)),
            "kw4": torch.ones(3, 4),
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureSerDer  # we don't save placeholder metadata
    @testing.expectedFailureNonStrict
    @testing.expectedFailureTrainingIRToRunDecompNonStrict  # source_fn_stack failure
    @testing.expectedFailureRetraceabilityNonStrict
    def test_linear_conv(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                x_conv = self.conv(x)
                x_linear = self.linear(x_conv)
                return x_linear.cos()

        ep = export(Foo(), (torch.randn(20, 16, 50, 100),))
        for node in ep.graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in ep.graph_signature.inputs_to_buffers
                or node.name in ep.graph_signature.inputs_to_parameters
            ):
                self.assertTrue("source_fn_stack" in node.meta)

    def test_export_api_with_dynamic_shapes(self):
        from torch.export import Dim, dims, export

        # pass dynamic shapes of inputs [args]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={k: {0: batch} for k in ["x", "y"]},
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        efoo = export(
            foo, inputs, kwinputs, dynamic_shapes={k: {0: batch} for k in ["x", "y"]}
        )
        self.assertEqual(
            efoo.module()(*inputs, **kwinputs).shape, foo(*inputs, **kwinputs).shape
        )

        # pass dynamic shapes of inputs [partial, error]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(batch\\)!(.*\n)*.*"
                "batch was inferred to be a constant(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = 10"
            ),
        ):
            export(
                foo,
                inputs,
                kwinputs,
                dynamic_shapes={"x": {0: batch}, "y": None},
            )

        # pass dynamic shapes of inputs [module]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [bounds, mostly shared]
        foo = Foo()
        inputs = (torch.randn(10, 3, 3), torch.randn(10, 3, 3))
        batch = Dim("batch", min=8, max=64)
        size = Dim("size")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={
                "x": (batch, size, size),
                "y": (batch, size, size),
            },
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s1])", "torch.Size([s0, s1, s1])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [multiple, mostly distinct]
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        efoo = export(
            Foo(),
            inputs,
            dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, s1, s2])", "torch.Size([s0, s2, s5])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dict]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs["x"], inputs["y"])

        foo = Foo()
        inputs = ({"x": torch.randn(10, 2, 3), "y": torch.randn(10, 3, 4)},)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": {k: {0: batch} for k in ["x", "y"]}}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [list]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs[0], inputs[1])

        foo = Foo()
        inputs = ([torch.randn(10, 2, 3), torch.randn(10, 3, 4)],)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": [{0: batch} for _ in range(2)]}
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dataclass]

        # TODO(avik): This part of the test should have failed both serde and retracing
        # but these failures are hidden because of the local import of `export` in this test.
        # The serde failure is benign, and easily avoided by moving the dataclass definition
        # to the top-level. OTOH the retracing failure needs further investigation.
        @dataclass
        class DataClass:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            DataClass,
            serialized_type_name="test_export_api_with_dynamic_shapes.DataClass",
        )

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs.a, inputs.b)

        foo = Foo()
        inputs = (DataClass(a=torch.randn(10, 2, 3), b=torch.randn(10, 3, 4)),)
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"inputs": [{0: batch}, {0: batch}]},
        )
        self.assertEqual(
            [
                str(node.meta["val"].shape)
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([s0, 2, 3])", "torch.Size([s0, 3, 4])"],
        )

        # pass dynamic shapes of inputs [pytree-registered classes]
        if HAS_TORCHREC:
            # skipping tests if torchrec not available
            class Foo(torch.nn.Module):
                def forward(self, kjt) -> torch.Tensor:
                    return kjt.values() + 0, kjt.offsets() + 0

            foo = Foo()
            kjt = KeyedJaggedTensor(
                values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                keys=["index_0", "index_1"],
                lengths=torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3]),
                offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
            )
            inputs = (kjt,)
            dim = Dim("dim")
            dim_plus_one = Dim("dim_plus_one")
            efoo = torch.export.export(
                foo,
                inputs,
                dynamic_shapes={"kjt": [{0: dim}, None, {0: dim}, {0: dim_plus_one}]},
            )
            self.assertEqual(
                [out.shape for out in efoo.module()(*inputs)],
                [out.shape for out in foo(*inputs)],
            )

        # pass dynamic shapes of inputs [distinct, error]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, K2, N = dims("batch", "M", "K1", "K2", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K2\\)!(.*\n)*.*"
                "K2.*and.*K1.*must always be equal(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K2 = K1"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, K2, N)},
            )

        # pass dynamic shapes of inputs [specialized, error]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, N = dims("batch", "M", "K1", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K1\\)!(.*\n)*.*"
                "K1 was inferred to be a constant(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K1 = 3"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, None, N)},
            )

        # pass dynamic shapes of inputs [guards, error]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] < 16 and y.shape[1] % 3 == 0:
                    return torch.matmul(x, y)
                else:
                    return x + y

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Not all values of K.*satisfy the generated guard(.*\n)*.*"
                "Not all values of batch.*satisfy the generated guard(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = Dim\\('batch', max=15\\)(.*\n)*.*"
                "K = 3\\*_K"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
            )

    def test_suggested_fixes_new_roots(self):
        from torch.export import dims

        # suggested fixes should introduce new root dim for modulo guard
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                # dy = 3 * _dx
                # dx = 3 * _dx - 1
                # dz = 3 * _dx + 2
                # suggested fixes results will look something like
                # {"dx": {"eq": 3*_dx-1, "min": 5, "max": 36}, "dy": {"eq": dx+1}, ...}
                if x.shape[0] >= 5 and x.shape[0] <= 36 and y.shape[0] % 3 == 0:
                    return x + y[1:] + z[3:]

        foo = Foo()
        inputs = (
            torch.randn(
                11,
            ),
            torch.randn(
                12,
            ),
            torch.randn(
                14,
            ),
        )
        dx, dy, dz = dims("dx", "dy", "dz")
        dynamic_shapes = {
            "x": (dx,),
            "y": (dy,),
            "z": (dz,),
        }
        with self.assertRaisesRegex(  # figure out regex later
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                "_dx = Dim\(\\'_dx\\', max=12\)(.*\n)*.*"
                "dx = 3\*_dx - 1(.*\n)*.*"
                "dy = 3\*_dx(.*\n)*.*"
                "dz = 3\*_dx \+ 2"
            ),
        ):
            export(Foo(), inputs, dynamic_shapes=dynamic_shapes)
        # retry export
        _dx = Dim("_dx", min=2, max=12)
        dynamic_shapes = {"x": (3 * _dx - 1,), "y": (3 * _dx,), "z": (3 * _dx + 2,)}
        export(Foo(), inputs, dynamic_shapes=dynamic_shapes)

    def test_refine_dynamic_shapes_from_suggested_fixes(self):
        from torch.export.dynamic_shapes import (
            refine_dynamic_shapes_from_suggested_fixes,
        )

        def helper(model, inputs, dynamic_shapes):
            # export, fail, parse & refine suggested fixes, re-export
            try:
                export(Foo(), inps, dynamic_shapes=dynamic_shapes)
                raise Exception("should have raised constraint violation error")
            except torch._dynamo.exc.UserError as exc:
                new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                    exc.msg, dynamic_shapes
                )
                export(Foo(), inps, dynamic_shapes=new_shapes)
                return new_shapes

        # specialize dims + derived dims
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                x0 = x + y[1:] + z[2:]
                x1 = x @ torch.randn(4, 4)
                return x0, x1

        inps = (
            torch.randn(
                4,
            ),
            torch.randn(
                5,
            ),
            torch.randn(
                6,
            ),
        )
        dx = Dim("dx", max=16)
        dynamic_shapes = {"x": (dx,), "y": (dx + 1,), "z": (dx + 2,)}
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0], 4)
        self.assertEqual(new_shapes["z"][0], 6)

        # refine lower, upper bound
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (torch.randn(16), torch.randn(12))
        dynamic_shapes = {"x": (Dim("dx"),), "y": (Dim("dy"),)}
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0].min, 6)
        self.assertEqual(new_shapes["y"][0].max, 16)

        # divisiblity, will introduce new root
        class Foo(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] >= 9:
                    return x.reshape([-1, 3])

        inps = (
            torch.randn(
                15,
            ),
        )
        dynamic_shapes = ((Dim("dx"),),)
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        dim = new_shapes[0][0]
        root = dim.root
        self.assertEqual(dim.fn(2), 6)
        self.assertEqual(root.min, 3)

        # turn dim into derived dim/relation
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[4:]

        inps = (torch.randn(6, 4), torch.randn(10, 4))
        dynamic_shapes = {
            "x": (Dim("dx0"), Dim("dx1")),
            "y": (Dim("dy0"), Dim("dy1")),
        }
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0], new_shapes["y"][0].root)  # dy0 = dx0 + 4
        self.assertEqual(new_shapes["y"][0].fn(5), 9)
        self.assertEqual(new_shapes["x"][1], new_shapes["y"][1])  # dx1 = dy1

        # nested dynamic shapes spec
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x0 = x[0]["data"] + x[1] + x[2][2:]
                x1 = y["a"] @ torch.randn(4, 4)
                x2 = y["b"] @ torch.randn(6, 6)
                return x0, x1, x2

        inps = (
            [
                {"data": torch.randn(4, 4)},
                torch.randn(4, 4),
                torch.randn(6, 4),
            ],
            {
                "a": torch.randn(8, 4),
                "b": torch.randn(9, 6),
            },
        )
        dynamic_shapes = {
            "x": [
                {"data": (Dim("dx00"), Dim("dx01"))},
                (Dim("dx10"), Dim("dx11")),
                (Dim("dx20"), Dim("dx21")),
            ],
            "y": {
                "a": (Dim("dya0"), Dim("dya1")),
                "b": (Dim("dyb0"), Dim("dyb1")),
            },
        }
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(
            new_shapes["x"][0]["data"][0], new_shapes["x"][1][0]
        )  # dx10 = dx00
        self.assertEqual(
            new_shapes["x"][2][0].root, new_shapes["x"][0]["data"][0]
        )  # dx20 = dx00 + 2
        self.assertEqual(new_shapes["x"][2][0].fn(10), 12)
        self.assertEqual(
            new_shapes["x"][0]["data"][1], new_shapes["x"][1][1]
        )  # dx11 = dx01
        self.assertEqual(new_shapes["y"]["a"][1], 4)
        self.assertEqual(new_shapes["y"]["b"][1], 6)
        self.assertEqual(new_shapes["y"]["b"][0].__name__, "dyb0")  # unchanged

    def test_dynamic_shapes_spec_with_pytree(self):
        from torch.export import Dim, export
        from torch.utils._pytree import tree_map

        inputs = {
            "tensor": torch.randn(3),
            "dict_of_tensors": {k: torch.randn(3) for k in ["A", "B", "C", "D"]},
            "list_of_tensors": [torch.randn(3) for _ in range(4)],
        }

        batch = Dim("batch")
        # uniformly specify dynamic shapes for all inputs
        spec = tree_map(lambda x: {0: batch}, inputs)

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return (
                    inputs["tensor"]
                    + inputs["dict_of_tensors"]["A"]
                    + inputs["list_of_tensors"][0]
                )

        ep = export(Foo(), (inputs,), dynamic_shapes={"inputs": spec})
        input_shapes = [
            str(node.meta["val"].shape)
            for node in ep.graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(len(input_shapes), 9)
        self.assertTrue(all(shape == "torch.Size([s0])" for shape in input_shapes))

    def test_error_does_not_reference_eager_fallback(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        fn_ddo = Module()
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
            error_msg = r"Could not guard on data-dependent expression"
        else:
            error = torchdynamo.exc.UserError
            error_msg = r"^(?!.*fall back to eager).*"
        with self.assertRaisesRegex(error, error_msg):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),))

    def test_pytree_register_data_class(self):
        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        dt = MyDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertTrue(spec, LeafSpec())
        self.assertTrue(len(flat) == 1)

        register_dataclass_as_pytree_node(
            MyDataClass,
            serialized_type_name="test_pytree_register_data_class.MyDataClass",
        )

        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(MyDataClass, [["x", "y"], ["z"]], [LeafSpec(), LeafSpec()]),
        )
        self.assertEqual(flat, [3, 4])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

        @dataclass
        class MyOtherDataClass:  # the pytree registration don't allow registering the same class twice
            x: int
            y: int
            z: int = None

        # Override the registration with keep none fields
        register_dataclass_as_pytree_node(
            MyOtherDataClass,
            return_none_fields=True,
            serialized_type_name="test_pytree_regster_data_class.MyOtherDataClass",
        )

        dt = MyOtherDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyOtherDataClass,
                [["x", "y", "z"], []],
                [LeafSpec(), LeafSpec(), LeafSpec()],
            ),
        )
        self.assertEqual(flat, [3, 4, None])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyOtherDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_register_nested_data_class(self):
        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)
        ab = Inner(3, 4)
        dt = Outer(xy, ab)
        inp = {"dt1": (dt, ({},)), "dt2": ((torch.ones(1),), dt)}

        register_dataclass_as_pytree_node(
            Inner, serialized_type_name="test_pytree_register_nested_data_class.Inner"
        )
        register_dataclass_as_pytree_node(
            Outer, serialized_type_name="test_pytree_register_nested_data_class.Outer"
        )

        flat, spec = tree_flatten(inp)
        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1), 1, 2, 3, 4])

        unflat = tree_unflatten(flat, spec)
        self.assertEqual(unflat, inp)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_param_util(self):
        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        ep = export(Basic(), (torch.randn(5, 10),))
        num_params = 0
        params = []
        for node in ep.graph.nodes:
            if is_param(ep, node):
                num_params += 1
                params.append(get_param(ep, node))
        self.assertEqual(num_params, 2)
        self.assertEqual(params[0].shape, [1, 10])  # weight
        self.assertEqual(params[1].shape, [1])  # bias

    def test_buffer_util(self):
        ep = export(
            torch.nn.BatchNorm2d(100, affine=False), (torch.ones(20, 100, 35, 45),)
        )
        num_buffer = 0
        buffer = []

        for node in ep.graph.nodes:
            if is_buffer(ep, node):
                num_buffer += 1
                buffer.append(get_buffer(ep, node))
        self.assertEqual(num_buffer, 3)

        self.assertEqual(buffer[0].shape, torch.Size([100]))  # running_mean
        self.assertEqual(buffer[1].shape, torch.Size([100]))  # running_var
        self.assertEqual(buffer[2].shape, torch.Size([]))  # num_batches_tracked

    def test_export_dynamo_config(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)

        config = DEFAULT_EXPORT_DYNAMO_CONFIG
        mod = MyModule()

        @contextmanager
        def _patch_config(kwargs):
            orig_config_dict = dataclasses.asdict(config)

            try:
                for k, v in kwargs.items():
                    setattr(config, k, v)
                yield
            finally:
                for k, v in orig_config_dict.items():
                    setattr(config, k, v)

        inp = (torch.rand(5, 4),)
        exported_program = export(mod, inp, strict=True)

        with _patch_config({"allow_rnn": False}):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "TorchDynamo purposely graph breaks on RNN, GRU, LSTMs",
            ):
                _ = export(mod, inp, strict=True)

    def test_device_to_static(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(Module(), (torch.tensor(1, device="cpu"),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    def test_device_to_dynamic(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(
            Module(),
            (torch.tensor([1, 2], device="cpu"),),
            dynamic_shapes={"x": {0: Dim("i")}},
        )
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    def test_device_to_mutation(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.to("cpu")
                y.add_(1)
                return y, x

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            export(Module(), (torch.tensor(1, device="cpu"),))

    def test_float_conversion(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        for op in ops:
            self.assertIn(op, (torch.ops.aten._to_copy.default,))

    def test_device_to_mutation_float(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.float()
                y.add_(1)
                return y, x

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            export(Module(), (torch.tensor(1, dtype=torch.float),))

    def test_module(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a, b = x
                a_conv = self.conv(a)
                a_linear = self.linear(a_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return (
                    a_linear.cos() + b_linear.sin(),
                    a_linear.sin() + b_linear.cos(),
                )

        inp_container = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[0], ep_rexported.module()(*inp_test)[0]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[1], ep_rexported.module()(*inp_test)[1]
            )
        )

    def test_use_embedding_twice(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(4, 4)

            def forward(self, x):
                return self.embed(x) + self.embed.weight[x]

        inputs = (torch.tensor([0, 1, 2, 3]),)
        ep = export(Foo(), inputs)

    def test_module_with_dict_container_inp_out(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a1, a2 = x["a"]
                b = x["b"]
                a1_conv = self.conv(a1)
                a1_linear = self.linear(a1_conv)
                a2_conv = self.conv(a2)
                a2_linear = self.linear(a2_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return {
                    "a": a1_linear.cos() + b_linear.sin(),
                    "b": a2_linear.sin() + b_linear.cos(),
                }

        inp_container = (
            {
                "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),
                "b": torch.randn(20, 16, 50, 100),
            },
        )

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = (
            {
                "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),
                "b": torch.randn(20, 16, 50, 100),
            },
        )

        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)["a"], ep_rexported.module()(*inp_test)["a"]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)["b"], ep_rexported.module()(*inp_test)["b"]
            )
        )

    def test_args_type_checked(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.rand(2, 2)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "to be a tuple"):
            # Intentionally not wrapping `inp` in a tuple to trigger the error
            _ = export(M(), inp)

    def test_decomp_item_in_prim_before_decomposition(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "Fail")
                return x

        ep = export(M(), (torch.randn(2, 2),))
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 1, exactly=True
        ).run(ep.graph_module.code)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_decomp_item_in_prim_after_decomposition(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "Fail")
                return x

        decomp_table = {**_decomp_table_to_post_autograd_aten(), **decomposition_table}

        ep = export(M(), (torch.randn(2, 2),)).run_decompositions(decomp_table)

        # The difference seems fine because export_for_training catches const tensor little differently.
        # Training IR produces:
        # graph():
        #     %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
        #     %x : [num_users=1] = placeholder[target=x]
        #     %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
        #     %detach_ : [num_users=1] = call_function[target=torch.ops.aten.detach_.default](args = (%lift_fresh_copy,), kwargs = {})
        #     %_assert_async : [num_users=0] = call_function[target=torch.ops.aten._assert_async.msg](args = (%detach_, Fail), kwargs = {})
        #     return (x,)
        #
        # Pre-dispatch functionalization produces:
        # graph():
        #     %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
        #     %x : [num_users=1] = placeholder[target=x]
        #     %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
        #     %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
        #     %_assert_async : [num_users=0] = call_function[target=torch.ops.aten._assert_async.msg](args = (%detach, Fail), kwargs = {})
        #     return (x,)
        #
        # Retracing:
        # graph():
        #     %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
        #     %x : [num_users=1] = placeholder[target=x]
        #     %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%c_lifted_tensor_0,), kwargs = {})
        #     %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%clone,), kwargs = {})
        #     %_assert_async : [num_users=0] = call_function[target=torch.ops.aten._assert_async.msg](args = (%detach, Fail), kwargs = {})
        #     return (x,)
        # The difference comes from the fact that prim has registration for aten.detach while not for aten.detach_.
        # The diference in retracing comes from the fact that we retrace at pre-dispatch level while the usual flow
        # traces to post-dispatch.
        if is_training_ir_test(self._testMethodName):
            self.assertExpectedInline(
                str(ep.graph_module.code).strip(),
                """\
def forward(self, c_lifted_tensor_0, x):
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    _assert_async = torch.ops.aten._assert_async.msg(lift_fresh_copy, 'Fail');  lift_fresh_copy = _assert_async = None
    return (x,)""",
            )
        elif is_retracebility_test(self._testMethodName):
            self.assertExpectedInline(
                str(ep.graph_module.code).strip(),
                """\
def forward(self, c_lifted_tensor_0, x):
    clone = torch.ops.prims.clone.default(c_lifted_tensor_0, memory_format = torch.preserve_format);  c_lifted_tensor_0 = None
    view_of = torch.ops.prims.view_of.default(clone);  clone = None
    view_of_1 = torch.ops.prims.view_of.default(view_of);  view_of = None
    view_of_2 = torch.ops.prims.view_of.default(view_of_1);  view_of_1 = None
    _assert_async = torch.ops.aten._assert_async.msg(view_of_2, 'Fail');  view_of_2 = _assert_async = None
    return (x,)""",
            )
        else:
            self.assertExpectedInline(
                str(ep.graph_module.code).strip(),
                """\
def forward(self, c_lifted_tensor_0, x):
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    view_of = torch.ops.prims.view_of.default(lift_fresh_copy);  lift_fresh_copy = None
    view_of_1 = torch.ops.prims.view_of.default(view_of);  view_of = None
    view_of_2 = torch.ops.prims.view_of.default(view_of_1);  view_of_1 = None
    _assert_async = torch.ops.aten._assert_async.msg(view_of_2, 'Fail');  view_of_2 = _assert_async = None
    return (x,)""",
            )

    def test_decomp_batch_norm_functional_predispatch(self):
        class ConvBatchnorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return (x,)

        mod = ConvBatchnorm()
        mod.eval()
        inp = torch.randn(1, 1, 3, 3)

        gm = torch.export._trace._export(mod, (inp,), pre_dispatch=True).module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    bn_num_batches_tracked = self.bn.num_batches_tracked;  bn_num_batches_tracked = None
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    _native_batch_norm_legit_no_training = torch.ops.aten._native_batch_norm_legit_no_training.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    getitem = _native_batch_norm_legit_no_training[0];  _native_batch_norm_legit_no_training = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )

        mod.train()
        gm_train = _export(mod, (inp,), pre_dispatch=True).module()
        self.assertExpectedInline(
            str(gm_train.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    bn_num_batches_tracked = self.bn.num_batches_tracked
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    add = torch.ops.aten.add.Tensor(bn_num_batches_tracked, 1)
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, True, 0.1, 1e-05);  conv2d = bn_weight = bn_bias = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy__default = torch.ops.aten.copy_.default(bn_running_mean, getitem_3);  bn_running_mean = getitem_3 = copy__default = None
    copy__default_1 = torch.ops.aten.copy_.default(bn_running_var, getitem_4);  bn_running_var = getitem_4 = copy__default_1 = None
    copy__default_2 = torch.ops.aten.copy_.default(bn_num_batches_tracked, add);  bn_num_batches_tracked = add = copy__default_2 = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )

    def test_constrain_size_in_eager(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.max().item()
                torch._check_is_size(n)
                return y + n

        fn = Module()
        ep = export(
            fn,
            (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
        self.assertTrue(torch.allclose(ep.module()(*test_inp), fn(*test_inp)))

    def test_constrain_size_with_constrain_value(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.max().item()
                torch._check(n >= 2)
                torch._check(n <= 10)
                torch._check_is_size(n)
                return y + n

        fn = Module()
        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = fn(torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))

        ep = export(
            fn,
            (torch.randint(3, 4, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for expression u[\d+] \>\= 2"
        ):
            test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
            _ = ep.module()(*test_inp)

    def test_constrain_size_with_various_cases(self):
        class Module1(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 0)
                return y.sum() + torch.ones(n, 5).sum()

        case1 = Module1()

        class Module2(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 0)
                torch._check(n <= 6)
                return y.sum() + torch.ones(n, 5).sum()

        case2 = Module2()

        class Module3(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 0)
                torch._check(n <= 1)
                return y.sum() + torch.ones(n, 5).sum()

        case3 = Module3()

        class Module4(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 2)
                return y.sum() + torch.ones(n, 5).sum()

        case4 = Module4()

        class Module5(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                torch._check(n >= 1)
                return y.sum() + torch.ones(n, 5).sum()

        case5 = Module5()

        ep = export(case1, (torch.tensor(1), torch.ones(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = case1(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                case1(torch.tensor(1), torch.ones(4, 5)),
            )
        )

        ep = export(case2, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case2(torch.tensor(7), torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case2(torch.tensor(9), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case2(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        _ = case3(torch.tensor(1), torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case4(torch.tensor(1), torch.randn(4, 5))

        ep = export(case4, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case4(torch.tensor(1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case4(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        ep = export(case5, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case5(torch.tensor(0), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case5(torch.tensor(5), torch.ones(4, 5)),
            )
        )

    def test_automatic_constrain_size(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                return y.sum() + torch.ones(n, 5).sum()

        ep = export(M(), (torch.tensor(1), torch.ones(4, 5)))

        # This is because we insert sym_constrain_range in the graph now
        error_msg = r"Invalid value range for -1 between"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = ep.module()(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                M()(torch.tensor(1), torch.ones(4, 5)),
            )
        )

    def test_cleanup_dynamic_markers(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                x, y = inputs["x"], inputs["y"]
                return x + y

        inputs = (
            {
                "x": torch.randn(4, 8),
                "y": torch.randn(4, 8),
            },
        )
        shapes = {
            "inputs": {
                "x": (Dim.AUTO, Dim.STATIC),
                "y": (Dim.DYNAMIC, Dim.STATIC),
            },
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        for tensor in inputs[0].values():
            for attr in [
                "_dynamo_weak_dynamic_indices",
                "_dynamo_dynamic_indices",
                "_dynamo_dynamic_range",
                "_dynamo_static_indices",
                "_dynamo_unbacked_indices",
            ]:
                self.assertFalse(hasattr(tensor, attr))

    def test_constrain_decomp(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.freq = torch.ones(5, 5)

            def forward(self, start_pos: torch.Tensor):
                pos = start_pos.item()
                torch._check_is_size(pos)
                torch._check(pos >= 0)
                torch._check(pos <= 4)
                return self.freq[pos] * self.freq[pos]

        ep = torch.export.export(M(), (torch.tensor(1),))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)

        decompose_ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_mixed_input(self):
        class Module(torch.nn.Module):
            def forward(self, a, b, alpha: int):
                return torch.add(a, b, alpha=alpha)

        func = Module()

        a = torch.rand(1, 2)
        b = torch.rand(1, 2)
        alpha = 10

        exported = export(func, (a, b, alpha))
        for node in exported.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(isinstance(node.meta["val"], (Tensor, int)))

    def test_tensor_constant_with_wrapped_method(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = torch.ones(4, 4)

            def forward(self, x):
                return x + self.constant, self.constant

        class Wrapper(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, *arg, **kwargs):
                return self.fn(*arg, **kwargs)

        inp = (torch.zeros(4, 4),)

        def test(m):
            m_result = m(*inp)
            ep_result = export(m, inp).module()(*inp)
            for m_t, ep_t in zip(m_result, ep_result):
                self.assertTrue(torch.allclose(m_t, ep_t))

        test(M())
        test(Wrapper(M().forward))

    def test_export_with_inline_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._check(a >= 4)
                torch._check(a <= 7)
                return torch.empty((a, 4))

        f = Module()
        ep = export(f, (torch.tensor([5]),))
        self.assertEqual(ep.module()(torch.tensor([6])).shape, (6, 4))

        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression u[\d+] \<\= 7",
        ) as cm:
            ep.module()(torch.tensor([30]))

    def test_export_with_inline_constraints_complex(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._check(a >= 4)
                torch._check(a <= 7)
                empty = torch.empty((a, 4))

                return torch.cat((empty.transpose(0, 1), torch.zeros(6, a)), 0)

        f = Module()
        ep = export(f, (torch.tensor([6]),))
        self.assertEqual(ep.module()(torch.tensor([5])).shape, (10, 5))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.ones(1), buffer))

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))

    def test_to_module_with_mutated_buffer_multiple(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                self.bar.buf.add_(2)
                bar = self.bar(x)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )

    def test_runtime_assert_for_prim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        foo = Foo()
        tensor_inp = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        exported = torch.export.export(
            foo, (tensor_inp, 5), dynamic_shapes=dynamic_shapes
        )
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(8, 5), 5), foo(torch.ones(8, 5), 5)
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5, but got 6"),
        ):
            _ = exported.module()(torch.ones(8, 5), 6)

        exported = torch.export.export(
            foo, (tensor_inp, 5.0), dynamic_shapes=dynamic_shapes
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5.0, but got 6.0"),
        ):
            _ = exported.module()(torch.ones(7, 5), 6.0)

    def test_runtime_assert_for_prm_str(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b, mode):
                return torch.div(a, b, rounding_mode=mode)

        foo = Foo()
        inps = (torch.randn(4, 4), torch.randn(4), "trunc")
        exported = export(foo, inps)
        with self.assertRaisesRegex(
            RuntimeError, "to be equal to trunc, but got floor"
        ):
            _ = exported.module()(torch.randn(4, 4), torch.randn(4), "floor")
        self.assertTrue(torch.allclose(exported.module()(*inps), foo(*inps)))

    def test_redundant_assert_max_upper_bound(self):
        class M(torch.nn.Module):
            def forward(self, x):
                b = x.nonzero()
                torch._check(b.shape[0] >= 3)
                return b

        m = M()
        inp = (torch.tensor([1, 1, 1, 0, 1]),)
        dim = torch.export.Dim("dim")
        ep = export(m, inp, dynamic_shapes=((dim,),))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer_multiple_update_sub_later(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )

    def test_retracable_ep(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        inp = torch.ones(5, 5)
        exported = torch.export.export(Foo(), (inp,))
        reexported = torch.export.export(exported.module(), (inp,))

        self.assertTrue(torch.allclose(Foo()(inp), reexported.module()(inp)))

        dim0_x = torch.export.Dim("dim0_x")
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes=({0: dim0_x},))
        reexported = torch.export.export(exported.module(), (inp,))
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be equal to 5, but got 7"
        ):
            reexported.module()(torch.ones(7, 5))

        reexported = torch.export.export(
            exported.module(), (inp,), dynamic_shapes=({0: dim0_x},)
        )
        self.assertTrue(
            torch.allclose(
                Foo()(torch.ones(7, 5)), reexported.module()(torch.ones(7, 5))
            )
        )

        # can't retrace with invalid inputs with respect to the original ExportedProgram
        dim0_x_v2 = torch.export.Dim("dim0_x_v2", min=3)
        exported_v2 = torch.export.export(
            Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x_v2}}
        )
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be >= 3, but got 2"),
        ):
            torch.export.export(exported_v2.module(), (torch.randn(2, 2),))

    def test_export_cond_symbool_pred(self):
        class A(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self):
                return self.buffer.cos()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = A()

            def forward(self, x):
                def true_fn(x):
                    return x.cos() + self.a().sum()

                def false_fn(x):
                    return x.sin()

                return cond(x.shape[0] > 4, true_fn, false_fn, [x])

        dim0 = torch.export.Dim("dim0", min=3)
        inp = torch.ones(6, 4)
        ep = export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0}})
        schema = get_hop_schema(ep)
        self.assertExpectedInline(
            str(schema),
            """cond(SymBool pred, GraphModule true_fn, GraphModule false_fn, Tensor[2] operands) -> Tensor[1]""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, b_a_buffer, x):
    sym_size_int_1 = torch.ops.aten.sym_size.int(x, 0)
    gt = sym_size_int_1 > 4;  sym_size_int_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [x, b_a_buffer]);  gt = true_graph_0 = false_graph_0 = x = b_a_buffer = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
        )
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(6, 4)), Foo()(torch.ones(6, 4)))
        )

    def test_aten_lift_fresh_copy(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.lift_fresh_copy(x)

        ep = export(M(), (torch.ones(6, 4),))
        found = False

        op = "torch.ops.aten.clone.default"
        FileCheck().check_count(op, 1, exactly=True).run(ep.graph_module.code)

    def test_cond_buffers(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                self.buffer = torch.nn.Buffer(torch.ones(2, 3) + 1)

            def true_fn(self, x):
                return x + self.param

            def false_fn(self, x):
                return x + self.buffer

            def forward(self, x):
                return cond(x.shape[0] == 4, self.true_fn, self.false_fn, [x])

        inp = torch.ones(2, 3)
        ep = torch.export.export(M(), (inp,))
        inp = torch.randn(2, 3)
        epm = ep.module()
        self.assertTrue(torch.allclose(epm(inp), M()(inp)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 1
            )

    # map_fn references module outside the module hierarchy
    @unittest.expectedFailure
    def test_map_buffers(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.tensor(5), requires_grad=False)
                )
                self.buffer = torch.nn.Buffer(torch.tensor(6) + 1)

        m1 = M1()

        def map_fn(x, y):
            z = x + y + m1.param + m1.buffer
            z.add_(4)
            return z

        class M(torch.nn.Module):
            def forward(self, xs, y):
                return map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2), torch.tensor(3))
        ep = torch.export.export(M(), example_inputs)
        example_inputs = (torch.randn(3, 2), torch.tensor(3))
        epm = ep.module()
        self.assertTrue(torch.allclose(epm(*example_inputs), M()(*example_inputs)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 2
            )

    def test_check_is_size_error(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                # We cannot automatically infer a is a size here because view
                # accepts -1
                return torch.randn(24).view(a, 4)

        f = Module()
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        else:
            error = torch._dynamo.exc.UserError
        error_msg = r"Could not guard on data-dependent expression"
        with self.assertRaisesRegex(error, error_msg):
            _ = export(f, (torch.tensor(6),))

    def test_train_eval_on_exported_preautograd_module(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                if x.shape[0] > 4:
                    return x.cos()
                return x.sin()

        graph_module = _export(Foo(), (torch.ones(7, 5),), pre_dispatch=True).module()
        with self.assertRaisesRegex(
            NotImplementedError, r"Calling train\(\) is not supported yet."
        ):
            graph_module.train()

        with self.assertRaisesRegex(
            NotImplementedError, r"Calling eval\(\) is not supported yet."
        ):
            graph_module.eval()

    def test_lifted_constants(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(3)

        f = Module()
        ep = export(f, (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 2)
        self.assertEqual(len(ep.constants), 1)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Foo(), (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 3)

        inp = (torch.tensor(5),)
        self.assertTrue(torch.allclose(ep.module()(*inp), Foo()(*inp)))

        transform = ep.run_decompositions()
        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertTrue(torch.allclose(ep.module()(*inp), transform.module()(*inp)))

    def test_tensor_attribute_zero_args(self):
        class Foo(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = torch.tensor(value)

            def forward(self):
                return self.x.clone()

        m = Foo([1, 2])
        ep = export(m, ())
        self.assertEqual(ep.graph_signature.lifted_tensor_constants, ["x"])

    def test_preserve_shape_dynamism_for_unused_inputs(self):
        @dataclass
        class Input:
            f: torch.Tensor
            p: torch.Tensor

        torch._export.utils.register_dataclass_as_pytree_node(
            Input,
            serialized_type_name="test_preserve_shape_dynamism_for_unused_inputs.Input",
        )

        class Module(torch.nn.Module):
            def forward(self, x: Input):
                return x.f + 1

        mod = Module()
        example_inputs = (Input(f=torch.ones(10, 4), p=torch.zeros(10, 4)),)
        ep_static = torch.export.export(mod, example_inputs)
        for node in ep_static.graph.nodes:
            if node.op == "placeholder":
                for s in node.meta["val"].shape:
                    self.assertIsInstance(s, int)

        dim0_x_f, dim0_x_p = torch.export.dims("dim0_x_f", "dim0_x_p")
        dynamic_shapes = {"x": [{0: dim0_x_f}, {0: dim0_x_p}]}
        ep_dynamic = torch.export.export(
            mod, example_inputs, dynamic_shapes=dynamic_shapes
        )
        for node in ep_dynamic.graph.nodes:
            if node.op == "placeholder":
                for i, s in enumerate(node.meta["val"].shape):
                    if i == 0:
                        self.assertIsInstance(s, torch.SymInt)
                    else:
                        self.assertIsInstance(s, int)

    def test_multiple_definitions_same_name_dim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        A = torch.export.Dim("C", min=3)
        B = torch.export.Dim("C", max=12)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Found different definitions Dim\\(.*min=3\\) and Dim\\(.*max=12\\) "
            "for the same symbolic dimension",
        ):
            torch.export.export(
                Foo(),
                (torch.randn(10, 10), torch.randn(10, 10)),
                dynamic_shapes={"x": (A, B), "y": (B, A)},
            )

    def test_export_with_wrong_inputs(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        exported_program = export(MyModule(), (torch.rand(2, 3),), {})
        with self.assertRaisesRegex(ValueError, "Trying to flatten user inputs"):
            exported_program.module()(torch.rand(2, 3), torch.rand(2, 3))

    def test_export_decomps_simple(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp)
        state_dict = ep.state_dict

        self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

        core_aten_ep = ep.run_decompositions()
        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))
        self.assertEqual(id(state_dict), id(ep.state_dict))

    @unittest.skipIf(IS_FBCODE, "We can't customize decomp in fbcode")
    def test_export_for_inference_e2e(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()

        decomp_table = torch.export.default_decompositions()

        def _custom_decomp_for_linear(x, weight, bias):
            return x + bias.sum()

        decomp_table[torch.ops.aten.linear.default] = _custom_decomp_for_linear
        del decomp_table[torch.ops.aten.sum.default]
        ep = torch.export.export_for_inference(
            m, inp, decomp_table=decomp_table, dynamic_shapes={"x": {0: Dim("batch")}}
        )

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, p_lin_weight, p_lin_bias, x):
    sum_1 = torch.ops.aten.sum.default(p_lin_bias);  p_lin_bias = None
    add = torch.ops.aten.add.Tensor(x, sum_1);  x = sum_1 = None
    return (add,)""",
        )

        ep_core = ep.run_decompositions()

        self.assertExpectedInline(
            str(ep_core.graph_module.code).strip(),
            """\
def forward(self, p_lin_weight, p_lin_bias, x):
    sum_1 = torch.ops.aten.sum.dim_IntList(p_lin_bias, []);  p_lin_bias = None
    add = torch.ops.aten.add.Tensor(x, sum_1);  x = sum_1 = None
    return (add,)""",
        )

        with self.assertRaisesRegex(RuntimeError, "Expected input"):
            ep.module()(torch.randn(4, 12))

        with self.assertRaisesRegex(RuntimeError, "Expected input"):
            ep_core.module()(torch.randn(4, 12))

    @unittest.skipIf(IS_FBCODE, "We can't customize decomp in fbcode")
    def test_export_decomp_torture_case_1(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp)

        def custom_decomp_callable(x, weight, bias):
            return x + bias

        decomp_table = default_decompositions()
        decomp_table[torch.ops.aten.linear.default] = custom_decomp_callable
        core_aten_ep = ep.run_decompositions(decomp_table)
        self.assertExpectedInline(
            str(core_aten_ep.graph_module.code).strip(),
            """\
def forward(self, p_lin_weight, p_lin_bias, x):
    add = torch.ops.aten.add.Tensor(x, p_lin_bias);  x = p_lin_bias = None
    return (add,)""",
        )

    @unittest.skipIf(IS_FBCODE, "We can't customize decomp in fbcode")
    def test_export_decomp_torture_case_2(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = export(Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50)))
        ep_has_linear_convd = ep.run_decompositions(decomp_table={})

        def _decompose_linear_custom(x, weight, bias):
            return torch.matmul(x, weight.T) + 2 * bias

        ep_decompose_linear = ep_has_linear_convd.run_decompositions(
            decomp_table={torch.ops.aten.linear.default: _decompose_linear_custom}
        )

        self.assertExpectedInline(
            str(ep_decompose_linear.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    matmul = torch.ops.aten.matmul.default(conv2d, permute);  conv2d = permute = None
    mul = torch.ops.aten.mul.Tensor(c_linear_bias, 2);  c_linear_bias = None
    add = torch.ops.aten.add.Tensor(matmul, mul);  matmul = mul = None
    cos = torch.ops.aten.cos.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(conv1d);  conv1d = None
    add_1 = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add_1,)""",
        )

    def test_export_decomps_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp, dynamic_shapes={"x": {0: Dim("batch")}})

        core_aten_ep = ep.run_decompositions()

        input_node = [
            node for node in core_aten_ep.graph.nodes if node.op == "placeholder"
        ][-1]
        self.assertTrue(isinstance(input_node.meta["val"].shape[0], torch.SymInt))

        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))

    def test_nonzero_2(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.nonzero(x)

        f = Module()
        ep = export(f, (torch.ones(2),))
        inp = torch.randn(2)
        self.assertTrue(torch.allclose(ep.module()(inp), torch.nonzero(inp)))

    def test_redundant_asserts(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = x.item()
                torch._check_is_size(y)
                return torch.zeros(y)

        f = Foo()

        ep = export(f, (torch.tensor([3]),))

        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

        ep = ep.run_decompositions()

        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_non_arg_name_dynamic_shapes_api(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a.sum() + b.sum()

        foo = Foo()
        dim = torch.export.Dim("dim")
        ep = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=(None, {0: dim}),
        )

        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        self.assertEqual(ep.module()(*test_inp), foo(*test_inp))

        ep_v2 = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=(None, None),
        )
        with self.assertRaisesRegex(
            RuntimeError, "shape\[0\] to be equal to 4, but got 7"
        ):
            ep_v2.module()(*test_inp)

    def test_constant_output(self):
        class ModuleConstant(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = torch.randn(3, 2)

            def forward(self):
                return self.b

        class ModuleNestedConstant(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bff = torch.randn(3, 2)

            def forward(self, x, y):
                return {"prediction": (x + y, self.bff)}

        mod = ModuleConstant()
        ep = export(mod, ())
        self.assertEqual(ep.module()(), mod())

        args = (torch.randn(3, 2), torch.randn(3, 2))
        mod = ModuleNestedConstant()
        ep = export(mod, args)
        self.assertEqual(ep.module()(*args), mod(*args))

    def test_non_arg_name_dynamic_shapes_api_with_kwarg(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b, kw1, kw2):
                return a.sum() + b.sum() + kw1.sum() - kw2.sum()

        foo = Foo()
        dim = torch.export.Dim("dim")
        dim_for_kw1 = torch.export.Dim("dim_for_kw1")
        ep = torch.export.export(
            foo,
            (torch.randn(4, 4), torch.randn(4, 4)),
            {"kw2": torch.ones(4, 4), "kw1": torch.zeros(4, 4)},
            # We are specifying dynamism on the first kwarg even though user passed in
            # different order
            dynamic_shapes=(None, {0: dim}, {0: dim_for_kw1}, None),
        )

        test_inp = (torch.randn(4, 4), torch.randn(7, 4))
        test_kwargs = {"kw2": torch.ones(4, 4), "kw1": torch.zeros(9, 4)}
        # This should work even if the kwarg order are flipped.
        self.assertEqual(
            ep.module()(*test_inp, **test_kwargs), foo(*test_inp, **test_kwargs)
        )

    def test_non_arg_name_dynamic_shapes_api_with_container_type(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return a[0].sum() + a[1].sum() + b.sum()

        inp_a = (torch.randn(4, 4), torch.randn(4, 4))
        inp_b = torch.randn(4, 4)
        inp = (inp_a, inp_b)

        count = 0

        def dynamify_inp(x):
            # Mark the second input a[1] dynamic
            nonlocal count
            if count == 1:
                dim = torch.export.Dim("dim", min=3)
                count += 1
                return {0: dim}
            count += 1
            return None

        dynamic_shapes = tree_map(dynamify_inp, inp)

        foo = Foo()
        ep = torch.export.export(foo, inp, dynamic_shapes=dynamic_shapes)

        test_inp = ((torch.randn(4, 4), torch.randn(2, 4)), torch.randn(4, 4))
        with self.assertRaisesRegex(RuntimeError, "shape\[0\] to be >= 3, but got 2"):
            ep.module()(*test_inp)

    def test_nested_module(self):
        class M1(torch.nn.Module):
            def forward(self, x):
                return x + x

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        ep = export(M2(), inps)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        add_nodes = [
            node
            for node in ep.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor
        ]
        self.assertEqual(len(add_nodes), 1)
        add_node = add_nodes[0]
        self.assertEqual(len(add_node.meta["nn_module_stack"]), 1)
        self.assertTrue("M2" in list(add_node.meta["nn_module_stack"].values())[0][1])

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=2] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %x), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    def test_nested_module_with_init_buffer(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = torch.ones(3, 3)

            def forward(self, x):
                return x + self.b

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        ep = export(M2(), inps)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 0)

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=2] = placeholder[target=x]
    %ones : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([3, 3],), kwargs = {device: cpu, pin_memory: False})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %ones), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    @testing.expectedFailureRetraceability  # Retracing tensor constants results in buffers
    def test_nested_module_with_constant_buffer(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = torch.tensor(5)

            def forward(self, x):
                return x + self.b

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        ep = export(M2(), inps)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 1)

        if is_training_ir_test(self._testMethodName):
            self.assertExpectedInline(
                str(ep.graph).strip(),
                """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %lift_fresh_copy), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
            )
        else:
            self.assertExpectedInline(
                str(ep.graph).strip(),
                """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %detach), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul,)""",
            )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    def test_nested_module_with_parameter(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(3, 3))
                self.b = torch.nn.Parameter(torch.tensor(5.0))

            def forward(self, x):
                return x + self.a * self.b

        class M2(torch.nn.Module):
            def forward(self, x):
                m = M1()
                return m(x) * x

        inps = (torch.randn(3, 3),)
        # Strict export segfaults (Issue #128109)
        ep = torch.export.export(M2(), inps, strict=False)
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 1)

        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %ones : [num_users=1] = call_function[target=torch.ops.aten.ones.default](args = ([3, 3],), kwargs = {device: cpu, pin_memory: False})
    %detach : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%ones,), kwargs = {})
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %detach_1 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%lift_fresh_copy,), kwargs = {})
    %detach_2 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_1,), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%detach, %detach_2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul_1,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    @testing.expectedFailureRetraceabilityNonStrict
    def test_lazy_module_kwargs(self):
        class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, *args, **kwargs):
                pass

            def forward(self, x, y):
                return x + y

        m = LazyModule()
        ep = export(m, (), {"x": torch.randn(3, 3), "y": torch.randn(3, 3)})
        inputs = {"x": torch.randn(3, 3), "y": torch.randn(3, 3)}
        self.assertEqual(ep.module()(**inputs), m(**inputs))

    def test_retrace_pre_autograd(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                self.buffer.add_(4)
                return x.sum() + self.buffer.sum()

        inp = torch.randn(4, 4)
        gm = export(
            Foo(),
            (inp,),
            dynamic_shapes=({0: torch.export.Dim("dim", min=3)},),
        ).module()

        with self.assertRaisesRegex(
            RuntimeError, escape("Expected input at *args[0].shape[0]")
        ):
            gm(torch.randn(2, 2))

        with self.assertRaisesRegex(
            RuntimeError, escape("Expected input at *args[0].shape[0]")
        ):
            export(gm, (torch.randn(2, 2),))

        ep = export(
            gm,
            (torch.randn(5, 4),),
            dynamic_shapes=({0: torch.export.Dim("dim", min=3)},),
        )

        test_inp = torch.ones(8, 4)
        self.assertTrue(torch.allclose(ep.module()(test_inp), Foo().forward(test_inp)))

    def test_runtime_assert_with_size(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                a = x.item()
                torch._check_is_size(a)
                torch._check(a <= y.size(0))
                return y[:a]

        ep = export(
            M(),
            (torch.tensor(5), torch.ones(10)),
            dynamic_shapes={"x": None, "y": {0: torch.export.Dim("t")}},
        )
        inp = (torch.tensor(6), torch.randn(13))
        self.assertTrue(torch.allclose(ep.module()(*inp), M()(*inp)))

    @unittest.skip("Test is only supposed to work with non-strict mode")
    def test_issue_113041(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(1.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.a

        def forward_hook(module: torch.nn.Module, inputs, output) -> torch.Tensor:
            return 2 * output

        seq = torch.nn.Sequential(TestModule()).eval()
        seq.b = torch.tensor(2)
        handle = seq.register_forward_hook(forward_hook)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = seq

            def forward(self, x):
                return self.seq(x) + self.seq.b

        inp = (torch.randn(2, 8),)
        ep = export(M(), inp)  # This errors because dynamo adds an extra input

    def test_export_with_fake_tensor_inputs(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        # Put the inputs on a device
        with fake_mode, torch.device("meta"):
            x = torch.rand(5, 2, 2)
            model = Model()

            exported_program = torch.export.export(model, (x,))
            export_res = exported_program.module()(x)
            exp_res = model(x)
            all_meta_val = [
                node.meta["val"]
                for node in exported_program.graph_module.graph.nodes
                if "val" in node.meta
            ]
            self.assertTrue(export_res.size() == exp_res.size())
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            self.assertTrue(
                all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val)
            )
            decomposed_ep = exported_program.run_decompositions()
            export_res = decomposed_ep.module()(x)
            self.assertTrue(export_res.size() == exp_res.size())

    @skipIfXpu
    def test_export_with_fake_tensor_inputs_on_cuda_devices(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                out = self.linear(x)
                return out

        # Put the inputs on a device
        with fake_mode, torch.device("meta"):
            x = torch.rand(5, 2, 2)
            model = Model()

        # Manualy set the fake_device of fake tensors.
        x.fake_device = torch.device("cuda:0")
        for n, p in model.named_parameters():
            p.fake_device = torch.device("cuda:0")

        # Need to set all the requires_grad of tensors to False, because fake_tensor with CUDA device
        # doesn't quite work well with aot_autograd right now due to some logic fails
        # the check in call getDeviceGuardImpl in InputMetadata.
        x.requires_grad = False
        for n, p in model.named_parameters():
            p.requires_grad = False

        def check_device_and_fake_mode():
            exported_program = torch.export.export(model, (x,))
            export_res = exported_program.module()(x)
            exp_res = model(x)
            all_meta_val = [
                node.meta["val"]
                for node in exported_program.graph_module.graph.nodes
                if "val" in node.meta
            ]
            self.assertTrue(export_res.size() == exp_res.size())
            self.assertTrue(all(val.device == x.device for val in all_meta_val))
            self.assertTrue(
                all(val.fake_mode is all_meta_val[0].fake_mode for val in all_meta_val)
            )

        check_device_and_fake_mode()

    def test_run_decomposition_supports_user_input_mutation(self):
        class SingleOp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.op = torch.ops.aten.native_batch_norm

            def forward(
                self,
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
                **kwargs,
            ):
                return self.op(
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                    **kwargs,
                )

        input = torch.randn(5, 5, 5)
        weight = torch.randn(5)
        bias = torch.randn(5)
        running_mean = torch.randn(5)
        running_var = torch.randn(5)
        training = True
        momentum = 0.5
        eps = 0.6

        model = SingleOp()
        output = model(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )

        ep = torch.export.export(
            model,
            args=(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            ),
        )
        ep.run_decompositions()
        self.assertEqual(
            ep.module()(
                input, weight, bias, running_mean, running_var, training, momentum, eps
            ),
            output,
        )

    def test_export_graph_with_no_inputs(self):
        # We saw this pattern when users want to export
        # a graph that initlizes the states of a model.
        class Module(torch.nn.Module):
            def forward(self):
                return torch.randn(3, 4), torch.randn(3, 4)

        f = Module()
        ep = torch.export.export(f, ())
        a, b = ep.module()()
        self.assertEqual(a.size(), torch.Size([3, 4]))
        self.assertEqual(b.size(), torch.Size([3, 4]))

        # Contains unbacked symint
        class M(torch.nn.Module):
            def forward(self):
                full = torch.full((), 11)
                i0 = full.item()
                return (torch.full((i0,), 0.0),)

        f = M()
        ep = torch.export.export(f, ())
        a = ep.module()()[0]
        self.assertEqual(a.size(), torch.Size([11]))
        self.assertEqual(a, torch.zeros(11))

    def test_pad_sequence(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch._C._nn.pad_sequence([x])

        m0 = Module()
        inputs = (torch.randn(3, 2),)
        ep = torch.export.export(
            m0, inputs, dynamic_shapes={"x": {0: Dim("batch_size")}}
        )
        self.assertEqual(ep.module()(*inputs), m0(*inputs))

        class ModuleBatchFirst(torch.nn.Module):
            def forward(self, x):
                return torch._C._nn.pad_sequence([x], batch_first=True)

        m1 = ModuleBatchFirst()
        inputs = (torch.randn(3, 2),)
        ep = torch.export.export(
            m1, inputs, dynamic_shapes={"x": {0: Dim("batch_size")}}
        )
        self.assertEqual(ep.module()(*inputs), m1(*inputs))

        class ModuleMulti(torch.nn.Module):
            def forward(self, x, y, z):
                return torch._C._nn.pad_sequence([x, y, z])

        m2 = ModuleMulti()
        inputs = (torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2))
        ep = torch.export.export(
            m2,
            inputs,
            dynamic_shapes={
                "x": {0: Dim("batch_size")},
                "y": {0: Dim("y")},
                "z": {0: Dim("z")},
            },
        )
        self.assertEqual(ep.module()(*inputs), m2(*inputs))

        class ModuleMultiBatchFirst(torch.nn.Module):
            def forward(self, x, y, z):
                return torch._C._nn.pad_sequence([x, y, z], batch_first=True)

        m3 = ModuleMulti()
        inputs = (torch.randn(5, 2), torch.randn(4, 2), torch.randn(3, 2))
        ep = torch.export.export(
            m2,
            inputs,
            dynamic_shapes={
                "x": {0: Dim("batch_size")},
                "y": {0: Dim("y")},
                "z": {0: Dim("z")},
            },
        )
        self.assertEqual(ep.module()(*inputs), m3(*inputs))

    def test_export_then_compile_tensor_ctor(self):
        class M(torch.nn.Module):
            def forward(self, scores, mask):
                scores = scores.masked_fill(
                    mask, torch.tensor(torch.finfo(scores.dtype).min)
                )  # (bs, n_heads, q_length, k_length)
                return scores

        tensor_cpu = torch.randn(2, 4)
        mask_cpu = torch.BoolTensor(
            [[False, True, False, False], [False, False, False, False]]
        )

        m = M().eval()
        # res_ref = m(tensor_cpu, mask_cpu)
        # print("res_ref is: {}".format(res_ref), flush=True)

        exported_model = _export(m, (tensor_cpu, mask_cpu), pre_dispatch=True).module()
        optimized_model = torch.compile(exported_model)
        optimized_model(tensor_cpu, mask_cpu)

    def test_export_input_mutation_static_shape(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x, y):
                x.view(3, 2, -1).add_(y)
                return x

        inputs = (torch.randn(12), torch.tensor(2))
        model = MutationModel()
        ep = export(model, inputs)
        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        self.assertEqual(ep.module()(*inputs_export), model(*inputs_model))
        self.assertEqual(inputs[0] + torch.tensor(2), inputs_model[0])
        self.assertEqual(inputs[0] + torch.tensor(2), inputs_export[0])

    def test_export_input_mutation_dynamic_shape(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x, y):
                x[0].mul_(y)
                return x

        inputs = ((torch.randn(12), torch.randn(3, 2)), 2.0)
        model = MutationModel()
        ep = torch.export.export(
            model,
            inputs,
            dynamic_shapes={"x": ({0: torch.export.Dim("dim")}, None), "y": None},
        )
        nodes = list(ep.graph.nodes)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertIsInstance(nodes[0].meta["val"], torch.Tensor)
        self.assertIsInstance(nodes[0].meta["val"].shape[0], torch.SymInt)

        inputs_export = copy.deepcopy(inputs)
        inputs_model = copy.deepcopy(inputs)
        self.assertEqual(ep.module()(*inputs_export), model(*inputs_model))
        self.assertEqual(inputs[0][0] * 2.0, inputs_model[0][0])
        self.assertEqual(inputs[0][0] * 2.0, inputs_export[0][0])

    def test_export_input_mutation_bug(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x[:, :2, :] = x[:, :2, :] + 1
                return x

        inputs = (torch.ones(4, 4, 4),)
        ep = torch.export.export(M(), inputs)
        m = ep.module()

        # Make the name conflict with a placeholder name that we get from
        # aot_export
        for i, node in enumerate(m.graph.nodes):
            if node.op == "placeholder":
                node.name = f"arg0_{i + 1}"
        m.recompile()

        ep = torch.export.export(m, inputs)

        inputs = (torch.randn(4, 4, 4),)
        self.assertEqual(
            ep.module()(*copy.deepcopy(inputs)), M()(*copy.deepcopy(inputs))
        )

    def test__scaled_dot_product_flash_attention(self):
        class Module(torch.nn.Module):
            def forward(self, q, k, v):
                res = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return res[0]

        m = Module()
        inputs = (
            torch.randn(5, 4, 3, 2),
            torch.randn(5, 4, 3, 2),
            torch.randn(5, 4, 3, 2),
        )
        ep = export(m, inputs)
        self.assertEqual(ep.module()(*inputs), m(*inputs))

    @testing.expectedFailureSerDer  # symfloat nyi
    def test_sym_sqrt(self):
        import math

        class M(torch.nn.Module):
            def forward(self, x):
                return x / torch.sym_sqrt(x.shape[0])

        ep = export(M(), (torch.ones(16, 4),), dynamic_shapes={"x": {0: Dim("dim")}})
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)
        FileCheck().check_count("torch._sym_sqrt", 1, exactly=True).run(
            ep.graph_module.code
        )

    def test_check_specialized_int(self):
        class SingleOp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.op = torch.ops.aten.scatter_add

            def forward(self, t, dim, index, src, **kwargs):
                return self.op(t, dim, index, src, **kwargs)

        t = torch.randn(10, 5)
        dim = -1
        index = torch.tensor(
            [
                [2, 4, 3, 1, 0],
                [0, 2, 1, 4, 3],
                [3, 1, 4, 2, 0],
                [4, 0, 3, 1, 2],
                [3, 0, 4, 1, 2],
            ]
        )
        src = torch.randn(5, 5)

        model = SingleOp()
        output = model(t, dim, index, src)

        ep = torch.export.export(model, args=(t, dim, index, src))
        ep = ep.run_decompositions()
        self.assertEqual(ep.module()(t, dim, index, src), output)

    def test_fqn(self):
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        test_inp = torch.randn(2, 3)

        torch_gm = _export_to_torch_ir(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            normalized_k = k.replace(".", "_")
            self.assertIn(normalized_k, torch_gm.state_dict())
            self.assertEqual(v, torch_gm.state_dict()[normalized_k])
        self.assertTrue(torch.allclose(torch_gm(test_inp), orig_eager(test_inp)))

        pre_autograd_gm = torch.export._trace._export(
            orig_eager, (torch.rand(2, 3),), {}, pre_dispatch=True
        ).module()
        for k, v in orig_eager.state_dict().items():
            normalized_k = k.replace(".", "_")
            self.assertIn(k, pre_autograd_gm.state_dict())
            self.assertEqual(v, pre_autograd_gm.state_dict()[k])
        self.assertTrue(torch.allclose(pre_autograd_gm(test_inp), orig_eager(test_inp)))

        ep = export(orig_eager, (torch.rand(2, 3),), {})
        for k, v in orig_eager.state_dict().items():
            # We do not need to normalize the key here because exported
            # program's state dict is able to contain the module information.
            self.assertIn(k, ep.state_dict)
            self.assertEqual(v, ep.state_dict[k])
        self.assertTrue(torch.allclose(ep.module()(test_inp), orig_eager(test_inp)))

    def test_nn_module_stack(self):
        class Leaf(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = Leaf()
                self.buffer = torch.nn.Buffer(torch.randn(4, 4))

            def forward(self, x):
                return self.buffer.sum() + self.leaf(x).sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                y = self.bar.buffer + x
                return (self.bar(x) + y.sum(),)

        inp = (torch.randn(4, 4),)
        mod = Foo()
        ep_strict = torch.export.export(mod, inp).run_decompositions()
        ep_non_strict = torch.export.export(mod, inp, strict=False).run_decompositions()

        gm_unflat_non_strict = unflatten(ep_non_strict)
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))

        gm_unflat_strict = unflatten(ep_strict)

        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%weight, [1, 0]), kwargs = {})
    %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%bias, %x, %permute), kwargs = {})
    return addmm""",
        )

        gm_flat_non_strict = ep_non_strict.module()
        gm_flat_strict = ep_strict.module()

        self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))

    def test_nn_module_stack_shared_submodule(self):
        class Leaf(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = Leaf()
                self.buffer = torch.nn.Buffer(torch.randn(4, 4))

            def forward(self, x):
                return self.buffer.sum() + self.leaf(x).sum()

        class BarDifferent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                a = self.leaf(x).sum()
                b = self.leaf(x).sum()
                return a + b

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()
                self.bar_different = BarDifferent()

            def forward(self, x):
                y = self.bar.buffer + x
                return (
                    self.bar(x) + self.bar_different(x + 2),
                    y.sum(),
                )

        inp = (torch.randn(4, 4),)
        mod = Foo()
        ep_strict = torch.export.export(mod, inp)
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        gm_unflat_non_strict = unflatten(ep_non_strict)
        self.assertTrue(hasattr(gm_unflat_non_strict, "bar"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "buffer"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar, "leaf"))
        self.assertTrue(hasattr(gm_unflat_non_strict.bar_different, "leaf"))

        gm_unflat_strict = unflatten(ep_strict)

        self.assertEqual(gm_unflat_non_strict(*inp), gm_unflat_strict(*inp))
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar.leaf.linear.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%x, %weight, %bias), kwargs = {})
    return linear""",
        )
        self.assertExpectedInline(
            str(gm_unflat_non_strict.bar_different.leaf.linear.graph).strip(),
            """\
graph():
    %add_2 : [num_users=1] = placeholder[target=add_2]
    %weight : [num_users=1] = get_attr[target=weight]
    %bias : [num_users=1] = get_attr[target=bias]
    %linear_1 : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%add_2, %weight, %bias), kwargs = {})
    return linear_1""",
        )

        gm_flat_non_strict = ep_non_strict.module()
        gm_flat_strict = ep_strict.module()

        self.assertEqual(gm_flat_non_strict(*inp), gm_flat_strict(*inp))

    def test_unflatten_no_unroll(self):
        inp = (torch.ones(1),)

        class N(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1) * 4
                self.buf = torch.nn.Buffer(torch.ones(1) * 4)

            def forward(self, x, b):
                if b:
                    return x + self.const + 1
                else:
                    return x + 2 * (self.buf + 1) - self.const

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x):
                x0 = x + 3
                x1 = self.n(x0, True)
                x2 = self.n(x0, False)
                return x1 + x2

        m = M()
        eager_result = m(*inp)

        def test(ep, swap):
            epm = ep.module()
            ufm = torch.export.unflatten(ep)

            exported_result = epm(*inp)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

            for fqn, mod in swap.items():
                ufm.set_submodule(fqn, mod)
            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

        if not is_retracebility_test(self._testMethodName):
            test(
                export(M(), inp, preserve_module_call_signature=("n",)),
                swap={"n": N(), "n@1": N()},
            )

        class _N(torch.nn.Module):
            def forward(self, x):
                return x + 5

        class _N_1(torch.nn.Module):
            def forward(self, x):
                return x + 6

        test(
            export(M(), inp),
            swap={"n": _N(), "n@1": _N_1()},
        )

    def test_preserve_module_call_signature_unflatten_specialization(self):
        class N(torch.nn.Module):
            def forward(self, x, b):
                if b:
                    return x + 1
                else:
                    return x + 2

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x):
                x0 = x + 3
                x1 = self.n(x0, True)
                return x1 + 4

        inp = (torch.ones(1),)
        m = M()
        eager_result = m(*inp)

        if not is_retracebility_test(self._testMethodName):
            ep = export(M(), inp, preserve_module_call_signature=("n",))
            epm = ep.module()
            ufm = torch.export.unflatten(ep)

            exported_result = epm(*inp)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

            ufm.set_submodule("n", N())
            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

    def test_unflatten_multiple_graphs_preserve_signature_no_error(self):
        class N(torch.nn.Module):
            def forward(self, x, b):
                if b:
                    return x + 1
                else:
                    return x + 2

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x):
                x = x + 3
                x = self.n(x, True)
                x = x + 4
                x = self.n(x, False)
                x = x + 5
                return x

        inp = (torch.ones(1),)
        m = M()
        eager_result = m(*inp)

        def test(ep):
            epm = ep.module()
            ufm = torch.export.unflatten(ep)

            exported_result = epm(*inp)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

        if not is_retracebility_test(self._testMethodName):
            test(export(M(), inp, preserve_module_call_signature=("n",)))

        test(export(M(), inp))

    @testing.expectedFailureRetraceabilityNonStrict
    def test_unflatten_multiple_graphs_state(self):
        class N(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(1), persistent=False)

            def forward(self, x, b):
                if b:
                    self.buf.add_(1)
                else:
                    self.buf.add_(2)
                return x + self.buf

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x):
                x = self.n(x, True)
                x = x + 1
                x = self.n(x, False)
                x = x + 1
                x = self.n(x, True)
                x = x + 1
                x = self.n(x, False)
                return x

        inp = (torch.ones(1),)
        m = M()
        eager_result = m(*inp)

        if not is_retracebility_test(self._testMethodName):
            with self.assertRaisesRegex(
                ValueError,
                r"Found multiple calls of module n that mutate buffer n.buf",
            ):
                # Unflattening while preserving signatures is NYI for this case.
                torch.export.unflatten(
                    export(M(), inp, preserve_module_call_signature=("n",))
                )

        ep = export(M(), inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)

        exported_result = epm(*inp)
        self.assertTrue(torch.allclose(exported_result, eager_result))

        unflattened_result = ufm(*inp)
        self.assertTrue(torch.allclose(unflattened_result, eager_result))

    def test_unflatten_multiple_graphs_shared_submodule(self):
        class N(torch.nn.Module):
            def forward(self, x, b):
                if b:
                    return x + 1
                else:
                    return x + 2

        def gen_m(n, n_1, p, p_1):
            # Create a module instance where self.n and self.p
            # share the same submodule instance.
            # The booleans n, n_1 and p, p_1 are passed to two calls each
            # to self.n and self.p, and they determine which path through
            # the shared submodule instance is taken during export.
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.n = N()
                    self.p = self.n

                def forward(self, x):
                    x = x + 3
                    x = self.n(x, n)
                    x = x + 4
                    x = self.n(x, n_1)
                    x = x + 5
                    x = self.p(x, p)
                    x = x + 6
                    x = self.p(x, p_1)
                    return x + 7

            return M()

        inp = (torch.ones(1),)

        def test(m, expected_graph, expected_fqns, expected_duplicates):
            eager_result = m(*inp)

            ep = export(m, inp)
            exported_result = ep.module()(*inp)
            # exported and eager results should match (baseline)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened = torch.export.unflatten(ep)
            unflattened_result = unflattened(*inp)
            # unflattened and eager results should match
            # (needs multiple specialized graphs for shared submodule instance)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

            # expected graph should call minimal number of specialized submodules
            self.assertExpectedInline(
                str(unflattened.graph).strip(),
                expected_graph,
            )

            # expected graph should contain minimal number of specialized submodule fqns
            self.assertEqual(
                sorted(
                    [
                        fqn
                        for fqn, _ in unflattened.named_modules(remove_duplicate=False)
                    ]
                ),
                expected_fqns,
            )
            # expected graph should contain minimal number of specialized submodule instances
            for a, b in expected_duplicates:
                if is_non_strict_test(self._testMethodName):
                    # NOTE: non-strict does not de-duplicate shared submodules through different fqns.
                    # In particular, we use different module ids for self.n and self.p calls in non-strict,
                    # but in strict we use the same module id, which enables additional reuse.
                    # This is pre-existing behavior that might need to be fixed orthogonally.
                    self.assertNotEqual(
                        id(getattr(unflattened, a)), id(getattr(unflattened, b))
                    )
                else:
                    self.assertEqual(
                        id(getattr(unflattened, a)), id(getattr(unflattened, b))
                    )

        test(
            gen_m(n=True, n_1=False, p=False, p_1=False),
            # p should share n_1 graph, p_1 should be optimized away
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, 3), kwargs = {})
    %n : [num_users=1] = call_module[target=n](args = (%add,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n, 4), kwargs = {})
    %n_1 : [num_users=1] = call_module[target=n@1](args = (%add_2,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n_1, 5), kwargs = {})
    %p : [num_users=1] = call_module[target=p](args = (%add_4,), kwargs = {})
    %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p, 6), kwargs = {})
    %p_1 : [num_users=1] = call_module[target=p](args = (%add_6,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p_1, 7), kwargs = {})
    return (add_8,)""",
            ["", "n", "n@1", "p"],
            [("n@1", "p")],
        )

        test(
            gen_m(n=True, n_1=False, p=True, p_1=False),
            # p should reuse n graph, p_1 should reuse n_1 graph
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, 3), kwargs = {})
    %n : [num_users=1] = call_module[target=n](args = (%add,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n, 4), kwargs = {})
    %n_1 : [num_users=1] = call_module[target=n@1](args = (%add_2,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n_1, 5), kwargs = {})
    %p : [num_users=1] = call_module[target=p](args = (%add_4,), kwargs = {})
    %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p, 6), kwargs = {})
    %p_1 : [num_users=1] = call_module[target=p@1](args = (%add_6,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p_1, 7), kwargs = {})
    return (add_8,)""",
            ["", "n", "n@1", "p", "p@1"],
            [("n", "p"), ("n@1", "p@1")],
        )

        test(
            gen_m(n=True, n_1=True, p=True, p_1=False),
            # n_1 should be optimized away, p should reuse n graph
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, 3), kwargs = {})
    %n : [num_users=1] = call_module[target=n](args = (%add,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n, 4), kwargs = {})
    %n_1 : [num_users=1] = call_module[target=n](args = (%add_2,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n_1, 5), kwargs = {})
    %p : [num_users=1] = call_module[target=p](args = (%add_4,), kwargs = {})
    %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p, 6), kwargs = {})
    %p_1 : [num_users=1] = call_module[target=p@1](args = (%add_6,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p_1, 7), kwargs = {})
    return (add_8,)""",
            ["", "n", "p", "p@1"],
            [("n", "p")],
        )

        test(
            gen_m(n=True, n_1=False, p=False, p_1=True),
            # p should reuse n_1 graph, p_1 should reuse n graph
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, 3), kwargs = {})
    %n : [num_users=1] = call_module[target=n](args = (%add,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n, 4), kwargs = {})
    %n_1 : [num_users=1] = call_module[target=n@1](args = (%add_2,), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%n_1, 5), kwargs = {})
    %p : [num_users=1] = call_module[target=p](args = (%add_4,), kwargs = {})
    %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p, 6), kwargs = {})
    %p_1 : [num_users=1] = call_module[target=p@1](args = (%add_6,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%p_1, 7), kwargs = {})
    return (add_8,)""",
            ["", "n", "n@1", "p", "p@1"],
            [("n", "p@1"), ("p", "n@1")],
        )

    def test_stack_trace(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                x = self.linear(x)
                x *= 2.0
                return x

        ep = export(
            Foo(),
            (torch.randn(4, 4),),
        )
        # check correct lines are in stack trace
        trace_mul = [node for node in ep.graph.nodes if node.name == "mul"][0].meta.get(
            "stack_trace", ""
        )
        self.assertTrue(
            re.search(r"test_export.py.*in forward\n.*x \*= 2.0", trace_mul)
        )
        trace_addmm = [
            node for node in ep.graph.nodes if node.name in ["addmm", "linear"]
        ][0].meta.get("stack_trace", "")
        self.assertTrue(
            re.search(
                r"test_export.py.*in forward\n.*x = self.linear\(x\)", trace_addmm
            )
        )

    def test_cond_with_module_stack_export_with(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_fn(x):
                    return self.linear(x).cos()

                def false_fn(x):
                    return self.linear(x).sin()

                return torch.cond(x.sum() > 4, true_fn, false_fn, [x])

        class CondExport(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x.cos() + self.bar(x)

        inp = (torch.randn(4, 4),)
        ep = torch.export.export(CondExport(), inp, strict=False)
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, p_bar_linear_weight, p_bar_linear_bias, x):
    cos = torch.ops.aten.cos.default(x)
    sum_1 = torch.ops.aten.sum.default(x)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [p_bar_linear_bias, p_bar_linear_weight, x]);  gt = true_graph_0 = false_graph_0 = p_bar_linear_bias = p_bar_linear_weight = x = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(cos, getitem);  cos = getitem = None
    return (add,)""",
        )
        schema = get_hop_schema(ep)
        self.assertExpectedInline(
            str(schema),
            """cond(Tensor pred, GraphModule true_fn, GraphModule false_fn, Tensor[3] operands) -> Tensor[1]""",
        )

        cond_top_level_nn_module_stack = [
            node.meta["nn_module_stack"]
            for node in ep.graph.nodes
            if node.name == "true_graph_0"
        ][0]

        self.assertTrue(
            "test_cond_with_module_stack_export_with.<locals>.Bar"
            in str(cond_top_level_nn_module_stack)
        )

    # TODO: See https://github.com/pytorch/pytorch/issues/115790
    @unittest.expectedFailure
    def test_cond_with_module_stack_export_with_unflatten(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_fn(x):
                    return self.linear(x).cos()

                def false_fn(x):
                    return self.linear(x).sin()

                return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        class CondExport(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x.cos() + self.bar(x)

        inp = (torch.randn(4, 4),)
        ep = torch.export.export(CondExport(), inp, strict=False)

        cond_top_level_nn_module_stack = [
            node.meta["nn_module_stack"]
            for node in ep.graph.nodes
            if node.name == "true_graph_0"
        ][0]

        # we can't preserve nn_module_stack for the subgraphs for now.
        for node in ep.graph_module.true_graph_0.graph.nodes:
            self.assertEqual(
                node.meta["nn_module_stack"], cond_top_level_nn_module_stack
            )

        # this doesn't work today
        gm_unflat_strict = unflatten(ep)

    def test_predispatch_cond(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pred = torch.nn.Buffer(torch.tensor(False))
                self.t = torch.nn.Buffer(torch.tensor(10))

            def forward(self, x, y):
                def true_fn(x, y):
                    with torch.enable_grad():
                        return x - 1 + self.t + y

                return torch.cond(
                    self.pred,
                    true_fn,
                    lambda x, y: x + 1 - self.t + y,
                    [x, y],
                )

        model = Model()
        with torch.no_grad():
            exported_program = torch.export.export_for_training(
                model,
                (torch.tensor(10), torch.tensor(12)),
                {},
                dynamic_shapes=None,
                strict=False,
            )

        schema = get_hop_schema(exported_program)
        self.assertExpectedInline(
            str(schema),
            """cond(Tensor pred, GraphModule true_fn, GraphModule false_fn, Tensor[3] operands) -> Tensor[1]""",  # noqa: B950
        )

        self.assertExpectedInline(
            str(exported_program.graph_module.code.strip()),
            """\
def forward(self, b_pred, b_t, x, y):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(b_pred, true_graph_0, false_graph_0, [b_t, x, y]);  b_pred = true_graph_0 = false_graph_0 = b_t = x = y = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
        )  # noqa: B950

        self.assertExpectedInline(
            str(exported_program.graph_module.true_graph_0.code.strip()),
            """\
def forward(self, b_t, x, y):
    submod_3 = self.submod_1
    add_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_3, x, b_t, y);  submod_3 = x = b_t = y = None
    getitem = add_1[0];  add_1 = None
    return (getitem,)""",
        )

        self.assertExpectedInline(
            str(exported_program.graph_module.true_graph_0.submod_1.code.strip()),
            """\
def forward(self, x, b_t, y):
    sub = torch.ops.aten.sub.Tensor(x, 1);  x = None
    add = torch.ops.aten.add.Tensor(sub, b_t);  sub = b_t = None
    add_1 = torch.ops.aten.add.Tensor(add, y);  add = y = None
    return (add_1,)""",
        )

    def test_predispatch_grad_wrappers(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                with torch.enable_grad():
                    x = x - y
                with torch.no_grad():
                    x = x + y
                return x

        # no grad
        model = Model()
        with torch.no_grad():
            ep_nograd = torch.export.export_for_training(
                model,
                (torch.tensor(10), torch.tensor(12)),
                {},
                dynamic_shapes=None,
                strict=False,
            )
        # check that only sub op is wrapped with grad_enabled
        getattr_nodes = [
            node for node in ep_nograd.graph.nodes if node.op == "get_attr"
        ]
        self.assertEqual(len(getattr_nodes), 1)
        grad_subgraph = getattr(ep_nograd.graph_module, getattr_nodes[0].target)
        op_node = [
            node for node in grad_subgraph.graph.nodes if node.op == "call_function"
        ][0]
        self.assertEqual(op_node.target._name, "aten::sub.Tensor")

        # enable grad
        model = Model()
        ep_grad = torch.export.export_for_training(
            model,
            (torch.tensor(10), torch.tensor(12)),
            {},
            dynamic_shapes=None,
            strict=False,
        )
        # check that only add op is wrapped with grad_enabled
        getattr_nodes = [node for node in ep_grad.graph.nodes if node.op == "get_attr"]
        self.assertEqual(len(getattr_nodes), 1)
        grad_subgraph = getattr(ep_grad.graph_module, getattr_nodes[0].target)
        op_node = [
            node for node in grad_subgraph.graph.nodes if node.op == "call_function"
        ][0]
        self.assertEqual(op_node.target._name, "aten::add.Tensor")

    @testing.expectedFailureRetraceability
    def test_layer_sharing(self):
        N, C, H, W = 1, 2, 2, 3

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layer = torch.nn.LayerNorm([C, H, W])
                self.norms = torch.nn.ModuleList(
                    [
                        layer,
                        layer,
                    ]
                )

            def forward(self, x):
                for norm in self.norms:
                    x = norm(x)
                return x

        m = Module()
        copied_m = copy.deepcopy(m)
        ep = export(copied_m, (torch.randn(N, C, H, W),))
        self.assertEqual(copied_m.state_dict(), m.state_dict())
        self.assertEqual(ep.state_dict, m.state_dict())

    def test_non_persistent_buffer(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.rand(2, 3), persistent=False)

            def forward(self, x):
                return self.foo + x

        class MyOuterModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = MyModule()

            def forward(self, x):
                return self.inner(x)

        inp = torch.rand(2, 3)

        def _test(m, non_persistent_buffer):
            ep = export(m, (inp,), {})

            self.assertEqual(ep.module()(inp), m(inp))
            # Non-persistent buffers should not show up in the state dict
            self.assertNotIn(non_persistent_buffer, ep.state_dict)
            named_buffers = {name: buffer for (name, buffer) in ep.named_buffers()}
            # But they should show up in named_buffers()
            self.assertIn(non_persistent_buffer, named_buffers)
            self.assertIn(non_persistent_buffer, ep.constants)
            self.assertEqual(len(ep.constants), 1)

            # Check the same properties of the unlifted module
            mod = ep.module()
            self.assertNotIn(non_persistent_buffer, mod.state_dict())
            mod_named_buffers = {name: buffer for (name, buffer) in mod.named_buffers()}
            self.assertIn(non_persistent_buffer, mod_named_buffers)
            self.assertIn(non_persistent_buffer, ep.constants)
            self.assertEqual(len(ep.constants), 1)
            self.assertEqual(mod(inp), m(inp))

        _test(MyModule(), "foo")
        _test(MyOuterModule(), "inner.foo")

    def test_export_with_set_grad_enabled(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                with torch.no_grad():
                    return self.linear(x)

        model = Model()
        ep = export(model, (torch.randn(4, 4),), {})
        # _export_for_traininig is using pre_dispatch=False
        # Therefore the set_grad calls are not replaced with a hop.
        if not is_training_ir_test(self._testMethodName):
            self.assertIn(
                "torch.ops.higher_order.wrap_with_set_grad_enabled",
                ep.graph_module.code,
            )
        gm = torch.export.export_for_training(model, (torch.randn(4, 4),)).module()
        self.assertIn(
            "set_grad_enabled",
            gm.code,
        )

    def test_export_with_autocast(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                with torch.autocast(
                    device_type="cuda", dtype=torch.int16, enabled=True
                ):
                    y = x.sin().sum()
                with torch.autocast(
                    device_type="cpu", dtype=torch.float16, enabled=True
                ):
                    z = y.sin().sum()
                return z

        model = Model()
        ep = export(model, (torch.randn(4, 4),), {})
        # autocast nodes do not exist after run_decomposition()
        if not is_training_ir_test(self._testMethodName):
            self.assertIn(
                "torch.ops.higher_order.wrap_with_autocast",
                ep.graph_module.code,
            )
        # _export_for_traininig is using pre_dispatch=False
        # Therefore the autocast calls are not replaced with a hop.
        gm = torch.export.export_for_training(model, (torch.randn(4, 4),)).module()
        self.assertIn(
            "autocast",
            gm.code,
        )

    def test_export_as_backend(self):
        def f(x, y):
            return x + y

        def my_custom_backend(gm, example_inputs):
            gm = (
                torch.export.export(gm, tuple(example_inputs), strict=False)
                .run_decompositions()
                .module()
            )
            return gm

        inp = (torch.randn(3, 3), torch.randn(3, 3))
        new_res = torch.compile(f, backend=my_custom_backend)(*inp)
        self.assertTrue(torch.allclose(f(*inp), new_res))

    def test_nonstrict_retrace_preserves_metadata(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        inp = torch.randn(4, 4)
        m = MyModule()
        ep = torch.export.export(m, (inp,), {}, strict=False)
        # retrace
        ep2 = torch.export.export(ep.module(), (inp,), {}, strict=False)

        for n1, n2 in zip(list(ep.graph.nodes), list(ep2.graph.nodes)):
            self.assertEqual(n1.meta.get("stack_trace"), n2.meta.get("stack_trace"))

    def test_fake_weights(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Parameter(torch.randn(4, 4))
                self.bar = torch.nn.Buffer(torch.randn(4, 4), persistent=False)
                self.baz = torch.nn.Buffer(torch.randn(4, 4), persistent=True)

            def forward(self, x):
                return self.foo + x + self.bar + self.baz

        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        with fake_mode:
            m = MyModule()
        inp = torch.randn(4, 4)
        ep = export(m, (inp,))
        # Can't compare outputs because the module has fake weights.

    def test_fake_inputs(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return self.foo + x

        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        m = MyModule()
        with fake_mode:
            inp = torch.randn(4, 4)

        ep = export(m, (inp,))
        self.assertEqual(ep.module()(torch.ones(4, 4)), m(torch.ones(4, 4)))

    def test_double_lifted_constants(self):
        class EmptyM(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                return (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))

        m = EmptyM()
        ep = torch.export.export(m, ())
        for out, real_out in zip(ep.module()(), m()):
            self.assertTrue(torch.allclose(out, real_out))

    def test_trace_under_fake(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return self.foo + x

        fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[])
        )
        with fake_mode:
            m = MyModule()
            inp = torch.randn(4, 4)
            # Can't use unqualified export() as it will attempt to deserialize
            # under a new FakeTensorMode.
            ep = torch.export.export(m, (inp,))

    def test_compiling_state(self):
        class TestModule1(torch.nn.Module):
            def forward(self, x):
                if torch._dynamo.is_compiling():
                    return x * 2
                else:
                    return x * 3

        class TestModule2(torch.nn.Module):
            def forward(self, x):
                if torch._utils.is_compiling():
                    return x * 2
                else:
                    return x * 3

        class TestModule3(torch.nn.Module):
            def forward(self, x):
                if torch.compiler.is_compiling():
                    return x * 2
                else:
                    return x * 3

        for m in [TestModule1(), TestModule2(), TestModule3()]:
            input = torch.randn(5)
            ep_strict = export(m, (input,), strict=True)
            ep_non_strict = export(m, (input,), strict=False)

            self.assertTrue(torch.allclose(input * 3, m(input)))
            self.assertTrue(torch.allclose(input * 2, ep_strict.module()(input)))
            self.assertTrue(torch.allclose(input * 2, ep_non_strict.module()(input)))

    def test_user_input_and_buffer_mutation(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.randn(4, 4))

            def forward(self, x):
                self.foo.add_(1)
                x.add_(1)
                return self.foo + x

        mod = MyModule()
        mod_copy = copy.deepcopy(mod)
        ep = export(mod_copy, (torch.rand(4, 4),))

        self.assertEqual(mod.foo, ep.module().foo)
        self.assertEqual(mod(torch.ones(4, 4)), ep.module()(torch.ones(4, 4)))

    def test_symint_tensor_return(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib.returns_tensor_symint(x)[0]

        self._test_export_same_as_eager(Module(), (torch.randn(4, 4),))

    def test_custom_op_auto_functionalize(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, z):
                return torch.ops.testlib.foo(x, z)

        inps = (torch.ones(5), torch.ones(5))
        inps_for_export = (torch.ones(5), torch.ones(5))
        inps_for_export_with_decomp = (torch.ones(5), torch.ones(5))

        ep = torch.export.export(M(), inps_for_export)
        x_new_eager, z_new_eager, legit_eager = M()(*inps)
        x_new_export, z_new_export, legit_export = ep.module()(*inps_for_export)
        self.assertTrue(torch.allclose(x_new_eager, x_new_export))
        self.assertTrue(torch.allclose(z_new_eager, z_new_export))
        self.assertTrue(torch.allclose(legit_eager, legit_export))

        ep = ep.run_decompositions()
        x_new_export, z_new_export, legit_export = ep.module()(
            *inps_for_export_with_decomp
        )
        self.assertTrue(torch.allclose(x_new_eager, x_new_export))
        self.assertTrue(torch.allclose(z_new_eager, z_new_export))
        self.assertTrue(torch.allclose(legit_eager, legit_export))

    def test_custom_op_auto_functionalize_pre_dispatch(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.foo_mutated(x)

        inps = (torch.ones(5),)

        ep = torch.export.export(M(), inps)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x)
    auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = x, z = cos);  x = cos = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_1 = torch.ops.aten.cos.default(getitem_3)
    return (getitem_3, getitem_3, cos_1)""",
        )

        ep = torch.export._trace._export(M(), inps, pre_dispatch=True)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x)
    auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = x, z = cos);  x = cos = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_1 = torch.ops.aten.cos.default(getitem_3)
    return (getitem_3, getitem_3, cos_1)""",
        )

    def test_custom_op_auto_warn_pre_dispatch(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.testlib.foo_functional(x)

        inps = (torch.ones(5),)

        ep = torch.export.export(M(), inps).run_decompositions()
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    cos = torch.ops.aten.cos.default(x)
    cos_1 = torch.ops.aten.cos.default(x);  x = None
    auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = cos, z = cos_1);  cos = cos_1 = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_2 = torch.ops.aten.cos.default(getitem_3);  getitem_3 = None
    return (cos_2,)""",
        )

        ep = torch.export._trace._export(M(), inps, pre_dispatch=True)
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, x):
    foo_functional = torch.ops.testlib.foo_functional.default(x);  x = None
    return (foo_functional,)""",
        )

    def test_placeholder_naming_collisions(self):
        # test collisions between nested user inputs
        class Foo(torch.nn.Module):
            def forward(self, x, x_foo, x_foo_0):
                return x["foo"][0] + x_foo[0] + x_foo_0

        inputs = (
            {"foo": [torch.randn(4, 4)]},
            (torch.randn(4, 4),),
            torch.randn(4, 4),
        )
        ep = export(Foo(), inputs)
        expected_names = ["x_foo_0", "x_foo_0_1", "x_foo_0_2"]
        real_names = [spec.arg.name for spec in ep.graph_signature.input_specs]
        self.assertEqual(expected_names, real_names)

        # test collisions between user inputs and params, buffers, constants
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(4))
                self.alpha = torch.nn.Buffer(torch.randn(4), persistent=True)
                self.beta = torch.nn.Buffer(torch.randn(4), persistent=False)
                self.gamma = torch.randn(4)

            def forward(self, p, b_alpha, b, c_gamma):
                p = p["param"] + self.param
                b = self.alpha + self.beta + b_alpha + b["beta"]
                c = self.gamma + c_gamma
                return p, b, c

        inputs = (
            {"param": torch.randn(4)},
            torch.randn(4),
            {"beta": torch.randn(4)},
            torch.randn(4),
        )
        ep = export(Foo(), inputs)
        expected_names = [  # user inputs should be prioritized, unprefixed
            ("p_param_1", InputKind.PARAMETER),
            ("b_alpha_1", InputKind.BUFFER),
            ("b_beta_1", InputKind.BUFFER),
            ("c_gamma_1", InputKind.CONSTANT_TENSOR),
            ("p_param", InputKind.USER_INPUT),
            ("b_alpha", InputKind.USER_INPUT),
            ("b_beta", InputKind.USER_INPUT),
            ("c_gamma", InputKind.USER_INPUT),
        ]
        real_names = [
            (spec.arg.name, spec.kind) for spec in ep.graph_signature.input_specs
        ]
        self.assertEqual(expected_names, real_names)

        # test collisions between user inputs & call_function nodes
        class Foo(torch.nn.Module):
            def forward(self, mul, add, add_1):
                return mul * mul + add * add_1

        ep = export(Foo(), (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)))
        expected_names_and_ops = [
            ("mul", "placeholder"),
            ("add", "placeholder"),
            ("add_1", "placeholder"),
            ("mul_1", "call_function"),
            ("mul_2", "call_function"),
            ("add_2", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [(node.name, node.op) for node in ep.graph.nodes]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)

    @skipIfCrossRef  # Dynamo changes the order of ops under Torch function modes
    def test_placeholder_naming_collisions_hoo_subgraphs(self):
        # test collisions between user inputs, top-level nodes, and HOO subgraph nodes
        class Foo(torch.nn.Module):
            def forward(self, x, mul, mul_1):
                _mul = x * x
                y = cond(
                    _mul.sum() > 0,
                    lambda x, y, z: x * y * z,
                    lambda x, y, z: x + y + z,
                    [_mul, mul, mul_1],
                )
                with torch.enable_grad():
                    y = y * y
                return y

        with torch.no_grad():
            ep = torch.export._trace._export(
                Foo(),
                (torch.randn(4), torch.randn(4), torch.randn(4)),
                pre_dispatch=True,
            )

        schema = get_hop_schema(ep)
        self.assertExpectedInline(
            str(schema),
            """cond(Tensor pred, GraphModule true_fn, GraphModule false_fn, Tensor[3] operands) -> Tensor[1]""",
        )
        # test cond subgraph
        expected_names_and_ops = [
            ("mul_2", "placeholder"),
            ("mul", "placeholder"),
            ("mul_1", "placeholder"),
            ("mul_3", "call_function"),
            ("mul_4", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [
            (node.name, node.op) for node in ep.graph_module.true_graph_0.graph.nodes
        ]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)
        # test set_grad_enabled subgraph
        expected_names_and_ops = [
            ("getitem", "placeholder"),
            ("mul_1", "call_function"),
            ("output", "output"),
        ]
        real_names_and_ops = [
            (node.name, node.op) for node in ep.graph_module.submod_1.graph.nodes
        ]
        self.assertEqual(expected_names_and_ops, real_names_and_ops)

        # test collisions between user inputs & higher order op subgraphs
        # (please never do this)
        class Foo(torch.nn.Module):
            def forward(self, input, true_graph, body_graph):
                x = input + true_graph[0] + true_graph[1]
                x = cond(x.sum() > 0, lambda x: x * 2.0, lambda x: x + 2.0, [x])
                x = cond(x.sum() > 0, lambda x: x * 2.0, lambda x: x + 2.0, [x])
                return x

        inputs = (
            torch.randn(10, 4),
            (torch.randn(4), torch.randn(4)),
            (torch.randn(4),),
        )
        ep = export(Foo(), inputs)
        expected_getattr_names = [
            "true_graph_2",
            "false_graph_0",
            "true_graph_3",
            "false_graph_1",
        ]
        real_getattr_names = [
            node.name for node in ep.graph.nodes if node.op == "get_attr"
        ]
        self.assertEqual(expected_getattr_names, real_getattr_names)

    def test_constant_input_naming(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, div="floor"):
                return torch.div(x, y, rounding_mode=div)

        f = Foo()
        inputs = (torch.randn(4), torch.randn(4), "floor")
        ep = export(f, inputs)
        div_spec = ep.graph_signature.input_specs[2]
        self.assertEqual(div_spec.arg.name, "div")
        self.assertEqual(div_spec.arg.value, "floor")

    def test_unbacked_deferred_runtime_retrace(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                y_sum = y.sin().sum()
                with torch.no_grad():
                    a = x.item()
                    torch._check_is_size(a)
                    torch._check(a > 2)
                    torch._check(a < 6)
                    unbacked_shape = torch.ops.testlib.foo_unbacked(a)
                return y + y_sum + unbacked_shape.sum()

        inps = (torch.tensor(4), torch.randn(5, 5))
        from torch.export import _trace

        ep_pre = _trace._export(Foo(), inps, pre_dispatch=True, strict=False)
        self.assertExpectedInline(
            str(ep_pre.graph_module.submod_1.code).strip(),
            """\
def forward(self, x):
    item = torch.ops.aten.item.default(x);  x = None
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(item);  sym_constrain_range_for_size_default = None
    ge_1 = item >= 3
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 3 on node 'ge_1'");  ge_1 = _assert_scalar_default = None
    le = item <= 5
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u1 <= 5 on node 'le'");  le = _assert_scalar_default_1 = None
    gt_1 = item > 2
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 2 < u1 on node 'gt_1'");  gt_1 = _assert_scalar_default_2 = None
    lt_1 = item < 6
    _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(lt_1, "Runtime assertion failed for expression u1 < 6 on node 'lt_1'");  lt_1 = _assert_scalar_default_3 = None
    foo_unbacked = torch.ops.testlib.foo_unbacked.default(item);  item = None
    return (foo_unbacked,)""",
        )
        ep_aot = ep_pre.run_decompositions()
        self.assertExpectedInline(
            str(ep_aot.graph_module.code).strip(),
            """\
def forward(self, x, y):
    sin = torch.ops.aten.sin.default(y)
    sum_1 = torch.ops.aten.sum.dim_IntList(sin, []);  sin = None
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x);  x = None
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense);  sym_constrain_range_for_size_default = None
    ge_1 = _local_scalar_dense >= 3
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u3 >= 3 on node 'ge_1'");  ge_1 = _assert_scalar_default = None
    le_1 = _local_scalar_dense <= 5;  _local_scalar_dense = None
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u3 <= 5 on node 'le_1'");  le_1 = _assert_scalar_default_1 = None
    full = torch.ops.aten.full.default([4, 4], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    add = torch.ops.aten.add.Tensor(y, sum_1);  y = sum_1 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(full, []);  full = None
    add_1 = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
    return (add_1,)""",
        )

    def test_nested_dynamic_shapes_spec(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                (a0, a1), (b0, b1), (c0, c1, c2) = x
                return a0 + a1 + b0 + b1 + c0 + c1 + c2

        f = Foo()
        inputs = (
            (1, 2),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
        )
        # make sure this gets parsed correctly as 7 individual inputs, not 3 tensors
        dynamic_shapes = {
            "x": (
                (None, None),
                (None, None),
                (None, None, None),
            )
        }
        export(f, (inputs,), dynamic_shapes=dynamic_shapes)

    @testing.expectedFailureRetraceabilityNonStrict
    def test_disable_forced_specializations_ok(self):
        # check that we don't force specialization, and defer to runtime asserts
        # with allow_complex_guards_as_runtime_asserts=True to successfully export
        # case 1: modulo guards
        from torch.export import dims

        class Mod4Reshape(torch.nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[0] - 1, 4, -1)  # Mod(s0*s1, 4*(s0-1)) = 0

        inputs = (torch.randn(10, 72),)
        dx, dy = dims("dx", "dy")
        ep = torch.export._trace._export(
            Mod4Reshape(),
            inputs,
            dynamic_shapes={"x": (dx, dy)},
            allow_complex_guards_as_runtime_asserts=True,
        )
        out1 = ep.module()(torch.randn(8, 7))
        self.assertEqual(out1.shape, torch.ones(7, 4, 2).shape)
        out2 = ep.module()(torch.randn(12, 11))
        self.assertEqual(out2.shape, torch.ones(11, 4, 3).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(Mod\(s0\*s1, 4\*s0 \- 4\), 0\) on node 'eq.*'",
        ):
            ep.module()(torch.randn(8, 8))  # fail

        # case 2: 2d reshape
        class FreeReshape(torch.nn.Module):
            def forward(self, x, y, z):
                return x.reshape([-1]) + y.reshape([-1]) + z  # s0*s1 = s2*s3 = s4

        inputs = (
            torch.randn(6, 8),
            torch.randn(3, 16),
            torch.randn(48),
        )
        dynamic_shapes = {
            "x": [Dim(f"dx{i}", min=2) for i in range(2)],
            "y": [Dim(f"dy{i}", min=2) for i in range(2)],
            "z": [Dim(f"dz{i}", min=4) for i in range(1)],
        }
        ep = torch.export._trace._export(
            FreeReshape(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            allow_complex_guards_as_runtime_asserts=True,
        )
        ep = export(FreeReshape(), inputs, dynamic_shapes=dynamic_shapes)
        out1 = ep.module()(torch.randn(48, 1), torch.randn(4, 12), torch.randn(48))
        self.assertEqual(out1.shape, torch.ones(48).shape)
        out2 = ep.module()(torch.randn(5, 8), torch.randn(4, 10), torch.randn(40))
        self.assertEqual(out2.shape, torch.ones(40).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\*s1, s2\*s3\) on node 'eq.*'",
        ):  # fail only at runtime
            ep.module()(torch.randn(5, 8), torch.randn(4, 5), torch.randn(30))  # fail

        # case 3: 3d reshape (previously failing with different issue)
        class Reshape3d(torch.nn.Module):
            def forward(self, x, y):
                return x.reshape([-1]) + y  # s0*s1*s2 = s3

        inputs = (
            torch.randn(4, 3, 2),
            torch.randn(24),
        )
        dynamic_shapes = {
            "x": (Dim("dx0", min=2), Dim("dx1", min=2), Dim("dx2", min=2)),
            "y": (Dim("dy", min=8),),
        }
        ep = torch.export._trace._export(
            Reshape3d(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            allow_complex_guards_as_runtime_asserts=True,
        )
        out1 = ep.module()(torch.randn(9, 7, 2), torch.randn(126))
        self.assertEqual(out1.shape, torch.ones(126).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\*s1\*s2, s3\) on node 'eq.*'",
        ):  # fail only at runtime
            ep.module()(torch.randn(4, 3, 2), torch.randn(10))  # fail

    def test_disable_forced_specializations_errors(self):
        # check error messages with hybrid symints
        class Foo(torch.nn.Module):
            def forward(self, w, x, y, z):
                return w.reshape([-1]) + x, y + z  # simple: s0*s1 = s2, s3 = s4

        inputs = (
            torch.randn(3, 4),
            torch.randn(12),
            torch.randn(4),
            torch.randn(4),
        )
        dynamic_shapes = {
            "w": [Dim(f"dw{i}") for i in range(2)],
            "x": [Dim(f"dx{i}") for i in range(1)],
            "y": [Dim("dy")],  # y & z incorrect, export is supposed to fail.
            "z": [Dim("dz")],  # suggested fix should be to match these up.
        }
        with self.assertRaisesRegex(  # if disable=True, suggested fixes should not specialize.
            torch._dynamo.exc.UserError,
            r".*Constraints violated(.*\n)*"
            r"Suggested fixes:(.*\n)*"
            r".*dz = dy(.*\n)*",
        ) as msg:
            export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )

    # TODO requires_grad doesn't seem to work with serialization.
    @testing.expectedFailureSerDer
    def test_preserve_requires_grad_placeholders(self):
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, x, y):
                return self.p + x + y

        m = Module()
        ep = export(m, (torch.randn(3, 3), torch.randn(3, 3, requires_grad=True)))
        placeholders = [
            node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
        ]
        self.assertTrue(placeholders[0].meta["val"].requires_grad)
        self.assertFalse(placeholders[1].meta["val"].requires_grad)
        self.assertTrue(placeholders[2].meta["val"].requires_grad)

    def test_reshape_view_helper(self):
        # see: https://github.com/pytorch/pytorch/issues/126607
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x = x.view(x.size(1), -1)
                # torch/_refs/__init__/_reshape_view_helper() will generate guards on reshape kernel(?)
                # Ne(s0, 20), so that reshape isn't no-op
                # Ne(Mod(s0, 20), 0), so that reshape needs to first flatten [s0, 20, 16] -> [s0*20, 16]
                # then split_dim -> [20, s0, 16]
                # check that these show up in graph
                return torch.nn.functional.softmax(
                    x, dim=0
                )  # don't think softmax actually creates any issues, just part of original test

        model = Model()
        x = torch.rand(1024, 20, 16)
        dynamic_shapes = {"x": {0: Dim("batch")}}
        ep = torch.export._trace._export(
            model,
            (x,),
            dynamic_shapes=dynamic_shapes,
            allow_complex_guards_as_runtime_asserts=True,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, 20\)",
        ):
            ep.module()(torch.randn(20, 20, 16))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(Mod\(s0, 20\), 0\)",
        ):
            ep.module()(torch.randn(400, 20, 16))
        ep.module()(torch.randn(42, 20, 16))

    def test_allow_explicit_guards_as_runtime_asserts(self):
        # check that explicit guards are treated as runtime assertions
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                # check that negation of first guard also shows up as runtime assertion
                if x.shape[0] == y.shape[0]:  # False
                    return x + y
                elif x.shape[0] == y.shape[0] ** 3:  # False
                    return x + 2, y + 3
                elif x.shape[0] ** 2 == y.shape[0] * 3:  # True
                    return x * 2.0, y * 3.0

        inputs = (torch.randn(6), torch.randn(12))
        dynamic_shapes = {"x": [Dim("dx", min=4)], "y": [Dim("dy", min=4)]}
        ep = torch.export._trace._export(
            Foo(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            allow_complex_guards_as_runtime_asserts=True,
        )
        # check forward pass
        out0, out1 = ep.module()(torch.randn(9), torch.randn(27))
        self.assertEqual(out0.shape, torch.ones(9).shape)
        self.assertEqual(out1.shape, torch.ones(27).shape)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, s1\)",
        ):  # fail only at runtime
            ep.module()(torch.randn(4), torch.randn(4))  # fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s0, s1\**3\)",
        ):
            ep.module()(torch.randn(64), torch.randn(4))  # fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s0\**2, 3\*s1\)",
        ):
            ep.module()(torch.randn(10), torch.randn(9))  # fail

        # this should be set with command line flag TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1,
        # but dynamo checks that at torch import time, so setting os.environ makes no difference
        # instead, manually patch dynamo config and test.
        # test that setting this flag removes runtime asserts
        from torch._dynamo import config as _dynamo_config

        with _dynamo_config.patch(
            do_not_emit_runtime_asserts=True,
        ):
            ep = torch.export._trace._export(
                Foo(),
                inputs,
                dynamic_shapes=dynamic_shapes,
                allow_complex_guards_as_runtime_asserts=True,
            ).run_decompositions()

        self.assertEqual(
            [
                node.target == torch.ops.aten._assert_scalar.default
                for node in ep.graph.nodes
            ].count(True),
            0,
        )

    def test_constant_output_dup(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = torch.ones(4, 4)

            def forward(self, x):
                return x + self.constant, self.constant

        ep = export(M(), (torch.ones(4, 4),)).run_decompositions()
        mod = ep.module()
        a, b = mod(torch.zeros(4, 4))
        self.assertTrue(torch.allclose(a, torch.ones(4, 4)))
        self.assertTrue(torch.allclose(b, torch.ones(4, 4)))

    def test_constant_requires_grad_const(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.randn(2, 2, requires_grad=True)

            def forward(self, x):
                return x.cos() + self.foo.sum()

        gm = export(M(), (torch.ones(2, 2),)).module()
        self.assertFalse(gm.foo.requires_grad)

    def test_constant_aliasing(self):
        class M1(torch.nn.Module):
            def __init__(self, m2, foo):
                super().__init__()
                self.m2 = m2
                self.foo = foo

            def forward(self, x):
                return x + self.foo + self.m2(x)

        class M2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.ones(3, 3, requires_grad=True)

            def forward(self, x):
                return x + self.foo

        m2 = M2()
        m1 = M1(m2, m2.foo)
        inps = (torch.ones(3, 3),)
        ep = export(m1, inps, strict=False)
        # check both constants appear in list
        self.assertEqual(sorted(list(ep.constants)), ["foo", "m2.foo"])
        # check only one input spec exists
        num_constant_inputs = [
            spec.kind == InputKind.CONSTANT_TENSOR
            for spec in ep.graph_signature.input_specs
        ].count(True)
        self.assertEqual(num_constant_inputs, 1)
        # unflatten
        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(m1(*inps), unflattened(*inps)))

    @testing.expectedFailureRetraceability
    def test_unused_aliases(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # param
                self.alpha = torch.nn.Parameter(torch.randn(4))
                self.beta = self.alpha
                self.gamma = self.alpha

            def forward(self, x):
                return x + self.gamma

        inps = (torch.randn(4),)
        ep = export(Foo(), inps)
        # placeholder nodes will be deduplicated in strict-mode,
        # but check that all params still appear in state dict
        for param in ["alpha", "beta", "gamma"]:
            self.assertTrue(param in ep.state_dict)

        # check that they also appear in unflattened state dict
        unep = unflatten(ep)
        for param in ["alpha", "beta", "gamma"]:
            self.assertTrue(param in unep.state_dict())

    @testing.expectedFailureRetraceabilityNonStrict
    def test_intermediate_shape_comp(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                z = torch.cat([x, x], dim=0)
                w = z.repeat(y.shape[0])
                return w.shape[0] + x.shape[0]

        inputs = (torch.randn(6), torch.randn(4))
        shapes = {
            "x": (Dim("dx0"),),
            "y": (Dim("dy"),),
        }
        ep = export(
            Foo(),
            inputs,
            dynamic_shapes=shapes,
        )
        # test that shape is from size compute, not sym_size call
        add_node = [node for node in ep.graph.nodes if node.target == operator.add][0]
        self.assertTrue(add_node.args[0].target == operator.mul)
        # test sym_size calls only happen on placeholders
        sym_size_nodes = [
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.sym_size.int
        ]
        self.assertEqual(len(sym_size_nodes), 2)
        self.assertTrue(
            all(node.args[0].op == "placeholder" for node in sym_size_nodes)
        )
        # dynamo will DCE the repeat node, AOTAutograd will leave it
        # training IR will also DCE due to retracing
        repeat_nodes = [
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.repeat.default
        ]
        self.assertEqual(
            len(repeat_nodes),
            1
            if is_non_strict_test(self._testMethodName)
            and not is_training_ir_test(self._testMethodName)
            else 0,
        )

    def test_checks_to_constrain_range(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                n = y.item()
                m = y.item()
                torch._check_is_size(n)
                torch._check(m >= 0)
                torch._check(n >= 3)
                torch._check(-m >= -9)  # m <= 9
                torch._check(n <= 6)
                # n has range [3, 9]
                return x[:n]

        inputs = (torch.randn(10), torch.tensor(6))
        ep = export(Foo(), inputs)
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)

        ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        ).run(ep.graph_module.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range_for_size.default", 1, exactly=True
        ).run(ep.graph_module.code)

        # check runtime
        ep.module()(torch.randn(10), torch.tensor(5))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression u[\d+] \>\= 3",
        ):
            ep.module()(torch.randn(10), torch.tensor(2))

    def test_cse_for_symint(self):
        class Foo(torch.nn.Module):
            # check sym ops only get computed once
            def forward(self, x, y):
                if (
                    x.shape[0] ** 2 - y.shape[0] ** 2 >= 4  # 16
                    and x.shape[0] ** 2 - y.shape[0] ** 2 <= 20
                    and x.shape[0] ** 2 - y.shape[0] ** 2 != 15
                ):
                    return x * 2, y * 2

        inputs = (torch.randn(5), torch.randn(3))
        shapes = {"x": (Dim("dx"),), "y": (Dim("dy"),)}
        ep = torch.export._trace._export(
            Foo(),
            inputs,
            dynamic_shapes=shapes,
            allow_complex_guards_as_runtime_asserts=True,
        )
        # count 2 pow nodes, 2 sym_size.int nodes
        self.assertEqual(
            [node.target for node in ep.graph.nodes].count(
                operator.pow,
            ),
            2,
        )
        FileCheck().check_count("torch.ops.aten.sym_size.int", 2, exactly=True).run(
            ep.graph_module.code
        )

        ep = ep.run_decompositions()
        self.assertEqual(
            [node.target for node in ep.graph.nodes].count(
                operator.pow,
            ),
            2,
        )
        FileCheck().check_count("torch.ops.aten.sym_size.int", 2, exactly=True).run(
            ep.graph_module.code
        )

    def test_slice_with_floordiv(self):
        # slice operation emits runtime assert s0//2 <= s1
        class M1(torch.nn.Module):
            def forward(self, x, y):
                d = x.size(0) // 2
                return y[d:]

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M1()

            def forward(self, x, y):
                d = x.size(0) // 2
                m1_res = self.m1(x, y)
                return y[d:] + m1_res

        inputs = (torch.ones(10), torch.ones(10))
        d0 = torch.export.Dim("d0", max=2048)
        d1 = torch.export.Dim("d1", max=2048)
        ep = export(
            M(),
            inputs,
            dynamic_shapes=((d0,), (d1,)),
        )
        ep.module()(torch.ones(8), torch.ones(4))
        ep.module()(torch.ones(8), torch.ones(5))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression \(s0//2\) \<\= s1",
        ):
            ep.module()(torch.ones(10), torch.ones(4))

    def test_split_const_gm_with_lifted_constants(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w_pre = torch.randn(4, 4)
                self.b = torch.randn(4)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4),)
        mod = Model()
        ep = torch.export.export(mod, example_inputs)
        new_gm = copy.deepcopy(ep.graph_module)
        new_sig = copy.deepcopy(ep.graph_signature)
        placeholder_nodes = [
            node for node in new_gm.graph.nodes if node.op == "placeholder"
        ]
        constants = {**ep.state_dict, **ep.constants}
        lifted_constants = {
            n.name: constants[spec.target]
            for n, spec in zip(placeholder_nodes, new_sig.input_specs)
            if spec.target is not None
        }
        const_gm, _ = split_const_gm(new_gm, lifted_constants)
        counter = 0
        for node in const_gm.graph.nodes:
            if node.op == "call_function":
                counter += 1
        self.assertTrue(counter > 0)
        test_input = torch.randn(4, 4)
        expected = new_gm(None, None, test_input)[0]
        actual = mod(test_input)
        self.assertEqual(actual, expected)
        const_gm, _ = split_const_gm(ep.graph_module, lifted_constants, lambda x: True)
        counter = 0
        for node in const_gm.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(False)

    def test_istft_op(self):
        class istft_class(torch.nn.Module):
            def forward(self, spec):
                window = torch.hann_window(1024).type(torch.FloatTensor)
                return torch.istft(
                    spec,
                    n_fft=1024,
                    hop_length=512,
                    window=window,
                    length=144000,
                )

        model = istft_class()
        real_part = torch.randn(1, 513, 282, dtype=torch.float32)
        imaginary_part = torch.randn(1, 513, 282, dtype=torch.float32)
        spec = torch.complex(real_part, imaginary_part)
        export(model, (spec,))

    def test_custom_op_preserve(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.testlib.foo_functional.default(x)
                return torch.ops.testlib.foo_mutated.default(y)

        decomp_table = torch.export.default_decompositions()

        # FIXME (We need to design a proper way that doesn't need _preserve_ops)
        ep = torch.export.export(M(), (torch.randn(4, 4),)).run_decompositions(
            decomp_table,
            _preserve_ops=(
                torch.ops.testlib.foo_functional.default,
                torch.ops.testlib.foo_mutated.default,
            ),
        )

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, x):
    foo_functional = torch.ops.testlib.foo_functional.default(x);  x = None
    cos = torch.ops.aten.cos.default(foo_functional)
    auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.testlib.foo.default, x = foo_functional, z = cos);  foo_functional = cos = None
    getitem_3 = auto_functionalized[3];  auto_functionalized = None
    cos_1 = torch.ops.aten.cos.default(getitem_3)
    return (getitem_3, cos_1)""",
        )

    def test_export_linear_preserve_dynamic_shape(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        mod = M()
        ep = export(
            mod,
            (torch.randn(8, 4),),
            dynamic_shapes={
                "x": {
                    0: Dim("x"),
                }
            },
        )

        if IS_FBCODE:
            ep = ep.run_decompositions(
                {}, _preserve_ops=(torch.ops.aten.linear.default,)
            )
        else:
            table = torch.export.default_decompositions()
            del table[torch.ops.aten.linear.default]
            ep = ep.run_decompositions(table)

        comp_mod = ep.module()
        inp1 = torch.randn(3, 4)
        inp2 = torch.randn(7, 4)
        self.assertTrue(torch.allclose(comp_mod(inp1), mod(inp1)))
        self.assertTrue(torch.allclose(comp_mod(inp2), mod(inp2)))

    @testing.expectedFailureRetraceabilityNonStrict
    def test_automatic_dynamic_shapes_simple_equality(self):
        # The next 3 test cases tests for automatic dynamic shapes specs, verifying that automatic dynamism
        # leads to replacement symbols being set for equalities, and inferred relationships being checked
        # with runtime asserts. Check that we specialize to static values when the program says so.
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        # case 1: direct equality between symbols
        class SimpleEquality(torch.nn.Module):
            def forward(self, x, y, z):
                # all inputs should have shape [s0, s1]
                return x + y + z

        inputs = tuple(torch.randn(6, 3) for _ in range(3))
        # fully dynamic
        self._check_dynamic_shapes_specs_and_shapes(
            SimpleEquality(),
            inputs,
            specs=[
                ((AUTO, AUTO), (AUTO, AUTO), (AUTO, AUTO)),
                [[AUTO, AUTO], [AUTO, AUTO], [AUTO, AUTO]],
                {"x": (AUTO, AUTO), "y": (AUTO, AUTO), "z": (AUTO, AUTO)},
            ],
            passing_shapes=[
                ((4, 4), (4, 4), (4, 4)),
                ((1, 1), (1, 1), (1, 1)),
                ((0, 9), (0, 9), (0, 9)),
            ],
            failing_shapes=[
                ((4, 4), (4, 4), (4, 3)),
                ((4, 4), (5, 4), (4, 5)),
            ],
            test_serdes=True,
        )
        # static s1
        self._check_dynamic_shapes_specs_and_shapes(
            # specifying just one dimension as static should be enough to specialize all s1
            SimpleEquality(),
            inputs,
            specs=[
                [{0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}, (AUTO, None)],
                {"x": (AUTO, AUTO), "y": (AUTO, AUTO), "z": (AUTO, None)},
            ],
            passing_shapes=[
                ((4, 3), (4, 3), (4, 3)),
                ((1, 3), (1, 3), (1, 3)),
                ((0, 3), (0, 3), (0, 3)),
            ],
            failing_shapes=[
                ((4, 4), (4, 4), (4, 4)),
                ((1, 1), (1, 1), (1, 1)),
                ((0, 9), (0, 9), (0, 9)),
            ],
            test_serdes=True,
        )
        # fully static
        self._check_dynamic_shapes_specs_and_shapes(
            # this should specialize all
            SimpleEquality(),
            inputs,
            specs=[{"x": (None, AUTO), "y": (AUTO, AUTO), "z": (AUTO, None)}],
            passing_shapes=[
                ((6, 3), (6, 3), (6, 3)),
            ],
            failing_shapes=[
                ((6, 4), (6, 4), (6, 4)),
                ((1, 3), (1, 3), (1, 3)),
                ((0, 9), (0, 9), (0, 9)),
            ],
            test_serdes=True,
        )

    @testing.expectedFailureRetraceabilityNonStrict
    def test_automatic_dynamic_shapes_constant_relation(self):
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        # case 2: related by constant: s0 + 4 = s1
        class OffBy4(torch.nn.Module):
            def forward(self, x, y):
                return x + y[4:]

        inputs = (torch.randn(6), torch.randn(10))
        # fully dynamic
        self._check_dynamic_shapes_specs_and_shapes(
            OffBy4(),
            inputs,
            specs=[
                ((AUTO,), (AUTO,)),
                {"x": (AUTO,), "y": (AUTO,)},
            ],
            passing_shapes=[
                ((10,), (14,)),
                ((3,), (7,)),
                ((2,), (6,)),
            ],
            failing_shapes=[
                ((10,), (13,)),
            ],
            test_serdes=True,
        )
        # static s1 should specialize s0
        self._check_dynamic_shapes_specs_and_shapes(
            OffBy4(),
            inputs,
            specs=[
                {"x": (AUTO,), "y": (None,)},
            ],
            passing_shapes=[
                ((6,), (10,)),
            ],
            failing_shapes=[
                ((10,), (14,)),
                ((3,), (7,)),
                ((2,), (6,)),
            ],
            test_serdes=True,
        )

    @testing.expectedFailureRetraceabilityNonStrict
    def test_automatic_dynamic_shapes_linear_relation(self):
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        # case 3: linear relation
        class LinearRel(torch.nn.Module):
            def forward(self, x, y):
                # x: [s0], y: [s1]
                # relation seems to be (s0 + 2) // 4 == s1
                return x[1::4] + y

        inputs = (torch.randn(21), torch.randn(5))

        # fully dynamic
        self._check_dynamic_shapes_specs_and_shapes(
            LinearRel(),
            inputs,
            specs=[
                ((AUTO,), (AUTO,)),
                {"x": (AUTO,), "y": (AUTO,)},
            ],
            passing_shapes=[
                ((33,), (8,)),
                ((32,), (8,)),
                ((31,), (8,)),
                ((30,), (8,)),
            ],
            failing_shapes=[
                ((34,), (8,)),
                ((22,), (5,)),
            ],
            test_serdes=False,
        )
        # static s1 shouldn't actually specialize s0 (guard: (s0 + 2) // 4 == 5)
        self._check_dynamic_shapes_specs_and_shapes(
            LinearRel(),
            inputs,
            specs=[
                ((AUTO,), None),
                {"x": (AUTO,), "y": None},
            ],
            passing_shapes=[
                ((21,), (5,)),
                ((20,), (5,)),
                ((19,), (5,)),
                ((18,), (5,)),
            ],
            failing_shapes=[
                ((33,), (8,)),
            ],
            test_serdes=False,
        )
        # but static s0 will definitely specialize s1 (guard: (21 + 2) // 4 == s1 -> 5 == s1)
        self._check_dynamic_shapes_specs_and_shapes(
            LinearRel(),
            inputs,
            specs=[
                (None, (AUTO,)),
            ],
            passing_shapes=[
                ((21,), (5,)),
            ],
            failing_shapes=[
                ((22,), (5,)),
            ],
            test_serdes=True,
        )

    def test_dynamic_shapes_serdes_generic(self):
        from torch._export.serde.dynamic_shapes import (
            _dump_dynamic_shapes,
            _load_dynamic_shapes,
        )

        class Foo(torch.nn.Module):
            def forward(self, a, b, c, d):
                if d == "hello":
                    x = a[0] + a[1][1:]
                    b = torch.cat([b, b], dim=0).reshape([-1, 1])
                    return x + b, c * 2

        # test de/serialization on some generic specs
        dz = Dim("dz", min=4, max=16)
        dx = 2 * dz
        dy = dx + 1
        inputs = (
            [
                torch.randn(8, 4),
                torch.randn(9, 4),
            ],
            torch.randn(4),
            torch.randn(4, 4),
            "hello",
        )
        dynamic_shapes = {
            "a": [
                (dx, 4),
                (dy, 4),
            ],
            "b": (dz,),
            "c": None,
            "d": None,
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes)
        self._check_dynamic_shapes_specs_and_shapes(
            Foo(),
            inputs,
            [dynamic_shapes],
            [
                ([(16, 4), (17, 4)], (8,), (4, 4), "hello"),
                ([(24, 4), (25, 4)], (12,), (4, 4), "hello"),
            ],
            [
                ([(16, 4), (17, 4)], (8,), (5, 5), "hello"),
            ],
            test_serdes=True,
        )
        self.assertExpectedInline(
            _dump_dynamic_shapes(dynamic_shapes, inputs),
            """DynamicShapesSpec(dynamic_shapes=([['2*dz', 4], ['2*dz + 1', 4]], ['dz'], ['_DimHint.STATIC', '_DimHint.STATIC'], None), dims={'dz': RootDim(min=4, max=16, derived=['2*dz', '2*dz + 1'])})""",
        )
        self.assertExpectedInline(
            _dump_dynamic_shapes(dynamic_shapes, inputs, to_dict=True),
            """{'dynamic_shapes': ([['2*dz', 4], ['2*dz + 1', 4]], ['dz'], ['_DimHint.STATIC', '_DimHint.STATIC'], None), 'dims': {'dz': {'min': 4, 'max': 16, 'derived': ['2*dz', '2*dz + 1']}}}""",
        )
        ((dx, _), (dy, _)), (dz,), (_, _), _ = _load_dynamic_shapes(
            _dump_dynamic_shapes(dynamic_shapes, inputs)
        )
        self.assertEqual(dx.root, dz)
        self.assertEqual(dy.root, dz)

    def test_dynamic_shapes_serdes_various(self):
        # serialization for dataclass inputs, Dim.AUTO/STATIC, and kwargs
        from torch._export.serde.dynamic_shapes import (
            _dump_dynamic_shapes,
            _load_dynamic_shapes,
        )

        auto, static = Dim.AUTO, Dim.STATIC

        @dataclass
        class Input:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            Input,
            serialized_type_name="test_dynamic_shapes_serdes_various.Input",
        )

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x - torch.randn(4), y.a + y.b + z[1:]

        args = (torch.randn(4, 4),)
        kwargs = {
            "y": Input(a=torch.randn(8, 8), b=torch.randn(8, 8)),
            "z": torch.randn(9, 8),
        }
        dynamic_shapes = {
            "x": (auto, static),
            "y": [(auto, auto), (auto, auto)],
            "z": (auto, 8),
        }

        # dump dynamic_shapes
        self.assertExpectedInline(
            _dump_dynamic_shapes(dynamic_shapes, args, kwargs),
            """DynamicShapesSpec(dynamic_shapes=(['_DimHint.AUTO', '_DimHint.STATIC'], [['_DimHint.AUTO', '_DimHint.AUTO'], ['_DimHint.AUTO', '_DimHint.AUTO']], ['_DimHint.AUTO', 8]), dims={})""",
        )
        self.assertExpectedInline(
            _dump_dynamic_shapes(dynamic_shapes, args, kwargs, to_dict=True),
            """{'dynamic_shapes': (['_DimHint.AUTO', '_DimHint.STATIC'], [['_DimHint.AUTO', '_DimHint.AUTO'], ['_DimHint.AUTO', '_DimHint.AUTO']], ['_DimHint.AUTO', 8]), 'dims': {}}""",
        )

    def test_dynamic_shapes_serdes_user_errors(self):
        # check error messages for dynamic shapes de/serialization
        from torch._export.serde.dynamic_shapes import (
            _dump_dynamic_shapes,
            _load_dynamic_shapes,
            DynamicShapesSpec,
            RootDim,
        )
        from torch._export.serde.serialize import _dataclass_to_dict

        # this stuff should be well tested in `test_mismatched_dynamic_shapes`
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: `inputs[0]['k']` "
                "is a <class 'list'>, but `dynamic_shapes[0]['k']` is a <class 'tuple'>"
            ),
        ):
            dynamic_shapes = {"x": {"k": (Dim("dx"), Dim("dy"))}}
            _dump_dynamic_shapes(dynamic_shapes, ({"k": [torch.randn(4, 4)]},))

        # loading with from_dict=True/False
        spec = DynamicShapesSpec(
            dynamic_shapes=[["dx"]],
            dims={"dx": RootDim(min=4, max=16, derived=[])},
        )
        spec_dict = _dataclass_to_dict(spec)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "With from_dict=True, expected `spec` to be a dict, "
                "got <class 'torch._export.serde.dynamic_shapes.DynamicShapesSpec'>"
            ),
        ):
            _load_dynamic_shapes(spec, from_dict=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape("Expected `spec` to be a DynamicShapesSpec, got <class 'dict'>"),
        ):
            _load_dynamic_shapes(spec_dict, from_dict=False)

        self.assertExpectedInline(
            _load_dynamic_shapes(spec, from_dict=False),
            """[[<class 'torch._export.serde.dynamic_shapes.dx'>]]""",
        )

        # check incorrect info in dims
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Expected dims in `spec['dims']` to map `min` to an int, got dx: None"
            ),
        ):
            spec = {
                "dynamic_shapes": [["dx"]],
                "dims": {
                    "dx": {
                        "min": None,
                        "max": 4,
                        "derived": [],
                    },
                },
            }
            _load_dynamic_shapes(spec, from_dict=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Expected dims in `spec['dynamic_shapes']` to be tracked in `spec['dims']`, "
                "got dx which is not in dict_keys(['dy'])"
            ),
        ):
            spec = {
                "dynamic_shapes": [["dx"]],
                "dims": {
                    "dy": {
                        "min": 2,
                        "max": 4,
                        "derived": [],
                    },
                },
            }
            _load_dynamic_shapes(spec, from_dict=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Expected derived expressions to be linear expressions, got dx**2 + 4"
            ),
        ):
            spec = {
                "dynamic_shapes": [["dx"]],
                "dims": {
                    "dx": {
                        "min": 2,
                        "max": 4,
                        "derived": ["dx**2 + 4"],
                    },
                },
            }
            _load_dynamic_shapes(spec, from_dict=True)

    @testing.expectedFailureSerDer  # TODO(pianpwk): PowByNatural valuerange deserialization
    @testing.expectedFailureRetraceabilityNonStrict
    def test_dim_dynamic(self):
        dynamic = Dim.DYNAMIC

        # dynamic should infer equalities and relations
        class Relations(torch.nn.Module):
            def forward(self, u, w, x, y, z):
                a = u[1:] + w + x  # s0 == s1 + 1 == s2 + 1
                b = y.flatten() + z  # s2*s3 == s4
                return a, b

        inputs = (
            torch.randn(5),
            torch.randn(4),
            torch.randn(4),
            torch.randn(4, 4),
            torch.randn(16),
        )
        ep = export(
            Relations(),
            inputs,
            dynamic_shapes={
                "u": (dynamic,),
                "w": (dynamic,),
                "x": (dynamic,),
                "y": (dynamic, dynamic),
                "z": (dynamic,),
            },
        )
        ep.module()(
            torch.randn(6),
            torch.randn(5),
            torch.randn(5),
            torch.randn(7, 8),
            torch.randn(56),
        )

        # dynamic should complain when force specialized
        class Specialize(torch.nn.Module):
            def forward(self, x):
                torch._check(x.shape[0] == 4)
                return x + 2

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Not all values of RelaxedUnspecConstraint.* are valid because .* was inferred to be a constant",
        ):
            ep = export(
                Specialize(),
                (torch.randn(4, 8),),
                dynamic_shapes={
                    "x": (dynamic, dynamic),
                },
            )

        # dynamic should handle complex guards in the same way as auto
        class ModConstraint(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)

        ep = export(
            ModConstraint(),
            (torch.randn(3, 4),),
            dynamic_shapes={
                "x": (dynamic, dynamic),
            },
        )
        ep.module()(torch.randn(5, 8))
        num_asserts = [
            node.target == torch.ops.aten._assert_scalar.default
            for node in ep.graph.nodes
        ].count(True)
        self.assertEqual(num_asserts, 1)
        with self.assertRaises(RuntimeError):
            ep.module()(torch.randn(4, 2))

    @testing.expectedFailureNonStrict
    @testing.expectedFailureTrainingIRToRunDecompNonStrict  # unbacked symint not tracked?
    @testing.expectedFailureSerDer  # T195866111
    @testing.expectedFailureRetraceabilityNonStrict
    def test_hints_wrapper(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x = x + y

                def inner_body_fn(x, y):
                    x = torch.relu(x)
                    x = x + y
                    return x

                def outer_body_fn(x, y):
                    x = hints_wrapper(
                        inner_body_fn, (x, y), {}, hints={"inner_body": True}
                    )
                    x = torch.abs(x)
                    return x

                res = hints_wrapper(
                    outer_body_fn, (x, y), {}, hints={"outer_body": True}
                )
                return res

        x = torch.randn(2, 4)
        y = torch.ones(4)

        ep = export(M(), (x, y))
        export_res = ep.module()(x, y)
        ref_res = M()(x, y)
        self.assertEqual(export_res, ref_res)
        self.assertExpectedInline(
            normalize_gm(ep.graph_module.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 4]", y: "f32[4]"):
        add: "f32[2, 4]" = torch.ops.aten.add.Tensor(x, y);  x = None

        hints_wrapper_body_graph_0 = self.hints_wrapper_body_graph_0
        hints_wrapper = torch.ops.higher_order.hints_wrapper(hints_wrapper_body_graph_0, (add, y), {}, hints = {'outer_body': True});  hints_wrapper_body_graph_0 = add = y = None
        getitem: "f32[2, 4]" = hints_wrapper[0];  hints_wrapper = None
        return (getitem,)

    class hints_wrapper_body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2, 4]", arg1_1: "f32[4]"):
            hints_wrapper_body_graph_0 = self.hints_wrapper_body_graph_0
            hints_wrapper = torch.ops.higher_order.hints_wrapper(hints_wrapper_body_graph_0, (arg0_1, arg1_1), {}, hints = {'inner_body': True});  hints_wrapper_body_graph_0 = arg0_1 = arg1_1 = None
            getitem: "f32[2, 4]" = hints_wrapper[0];  hints_wrapper = None
            abs_1: "f32[2, 4]" = torch.ops.aten.abs.default(getitem);  getitem = None
            return (abs_1,)

        class hints_wrapper_body_graph_0(torch.nn.Module):
            def forward(self, arg0_1: "f32[2, 4]", arg1_1: "f32[4]"):
                relu: "f32[2, 4]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
                add: "f32[2, 4]" = torch.ops.aten.add.Tensor(relu, arg1_1);  relu = arg1_1 = None
                return (add,)
""",
        )

    def test_export_for_training_with_state_dict_hooks(self):
        def _state_dict_pre_hook(mod, prefix, keep_vars):
            mod._buffers["test"] = torch.Tensor([1])

        def _state_dict_hook(mod, state_dict, prefix, *args, **kwargs):
            keys = list(state_dict.keys())
            for key in keys:
                local_key = key[len(prefix) :]
                if local_key.startswith("layer"):
                    new_key = prefix + local_key.replace("layer.", "")
                    state_dict[new_key] = state_dict[key]
                    if new_key != key:
                        del state_dict[key]

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._register_state_dict_hook(_state_dict_hook)
                self.register_state_dict_pre_hook(_state_dict_pre_hook)
                # non-persistent buffer in named_buffers()
                self.foo = torch.nn.Buffer(torch.rand(2, 3), persistent=False)
                # non-persistent buffer not in named_buffers()
                self.register_buffer("buf", None, persistent=False)
                self.layer = Layer()

            def forward(self, x):
                x = self.layer(x)
                return x

        M = CustomModule()
        inp = (torch.randn(2, 2),)
        ep = export(M, inp)
        export_res = ep.module()(*inp)
        ref_res = M(*inp)
        self.assertEqual(export_res, ref_res)
        # we want to store the unprocessed keys
        self.assertTrue(
            {
                "layer.linear1.weight",
                "layer.linear1.bias",
                "layer.linear2.weight",
                "layer.linear2.bias",
            }.issubset({spec.target for spec in ep.graph_signature.input_specs})
        )
        unflattened = torch.export.unflatten(ep)
        export_res = unflattened(*inp)
        self.assertEqual(export_res, ref_res)

        with torch._export.utils._disable_load_state_dict_hooks(M):
            state_dict = M.state_dict()
        self.assertEqual(
            {
                "layer.linear1.weight",
                "layer.linear1.bias",
                "layer.linear2.weight",
                "layer.linear2.bias",
            },
            state_dict.keys(),
        )
        state_dict = M.state_dict()
        self.assertEqual(
            {
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias",
                "test",
            },
            state_dict.keys(),
        )


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestOneOffModelExportResult(TestCase):
    def test_scaled_dot_product_attention_cpu(self):
        """
        This test makes sure we are always getting the same decomposition result for SDPA.
        As of now _scaled_dot_product_flash_attention_for_cpu is expected to show up in
        export() result. Some downstream backend then further decompose it into core ATen
        ops in torch/_decomp/decompositions.py (search for
        _scaled_dot_product_flash_attention_for_cpu).

        Export is decomposing based on the CompositeImplicitAutograd kernel implementation
        of SDPA. If this test fails, it means the kernel is being modified. In this case
        we strongly encourage you to change the decomposition rule under
        torch/_decomp/decompositions.py along with the kernel changes, so all of the
        downstream backends are not being affected.
        """

        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, None, dropout_p=0.0, is_causal=True
                )
                return attn_output

        q = torch.randn(1, 1, 8, 8, device="cpu")
        k = torch.randn(1, 1, 8, 8, device="cpu")
        v = torch.randn(1, 1, 8, 8, device="cpu")

        from torch.nn.attention import SDPBackend

        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
            ep = torch.export.export(ScaledDotProductAttention(), (q, k, v))
            print(ep.graph)
            ep.run_decompositions()
            print(ep.graph)

    #         self.assertExpectedInline(ep.graph_module.code.strip(), """\
    # def forward(self, arg0_1, arg1_1, arg2_1):
    #     _scaled_dot_product_flash_attention_for_cpu = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(arg0_1, arg1_1, arg2_1, 0.0, True);  arg0_1 = arg1_1 = arg2_1 = None
    #     getitem = _scaled_dot_product_flash_attention_for_cpu[0];  _scaled_dot_product_flash_attention_for_cpu = None
    #     return (getitem,)""")

    @skipIfCrossRef
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_scaled_dot_product_attention_cuda(self):
        """
        This test makes sure we are always getting the same decomposition result for SDPA.
        As of now _scaled_dot_product_flash_attention is expected to show up in
        export() result (GPU tensors are given). Currently there's no downstream
        backend relies on this export result so if this test fails, feel free to
        change it to the latest export() result.
        """

        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, None, dropout_p=0.0, is_causal=True
                )
                return attn_output

        q = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 16, 16, 64, dtype=torch.bfloat16, device="cuda")

        ep = torch.export.export(
            ScaledDotProductAttention(), (q, k, v)
        ).run_decompositions()
        code_str = """\
def forward(self, q, k, v):
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, True, scale = 0.125);  q = k = v = None
    getitem = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
    return (getitem,)"""
        if SM90OrLater and not torch.version.hip:
            code_str = """\
def forward(self, q, k, v):
    _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(q, k, v, None, False, 0.0, True);  q = k = v = None
    getitem = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None
    return (getitem,)"""
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            code_str,
        )

    def test_int_list_output(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return [((1, 3), [x + x, x * x])]

        ep = torch.export.export(M(), (torch.ones(2, 3),))
        res = ep.module()(torch.ones(2, 3))
        self.assertEqual(res[0][0], (1, 3))

    def test_primitive_constant_output(self):
        class Z(torch.nn.Module):
            def forward(self, x, y):
                with torch.no_grad():
                    return y * x, "moo"

        ep = torch.export.export(Z(), (torch.tensor(3), 5))
        res = ep.module()(torch.tensor(4), 5)
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], "moo")

        class B(torch.nn.Module):
            def forward(self, x, y):
                return y * x, y

        ep = torch.export.export(B(), (torch.tensor(3), 5))
        res = ep.module()(torch.tensor(4), 5)
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], 5)

        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1] to be equal to 5, but got 20"),
        ):
            res = ep.module()(torch.tensor(4), 20)

        class F(torch.nn.Module):
            def forward(self, x):
                # return a constant of primitive type
                y = 5
                return y * x, y

        ep = torch.export.export(F(), (torch.tensor(3),))
        res = ep.module()(torch.tensor(4))
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], 5)

        class Q(torch.nn.Module):
            def forward(self, x, y):
                return y * x, y - 1

        ep = torch.export.export(Q(), (torch.tensor(3), 5))
        res = ep.module()(torch.tensor(4), 5)
        self.assertEqual(res[0], torch.tensor(20))
        self.assertEqual(res[1], 4)

    def test_unbacked_sdpa(self):
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from torch.nn.functional import scaled_dot_product_attention

        class Module(torch.nn.Module):
            def forward(
                self, query: torch.Tensor, cache: torch.Tensor, start_pos: torch.Tensor
            ) -> torch.Tensor:
                # x.sizes(): 1, 128, 16, 128
                sp = start_pos.item()
                torch._check_is_size(sp)
                torch._check(sp >= 0)
                torch._check(sp <= 126)
                key = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
                value = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
                query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp#L732
                return scaled_dot_product_attention(query, key, value)

        cache = torch.randn(1, 128, 16, 128, dtype=torch.float16)
        query = torch.randn(1, 1, 16, 128, dtype=torch.float16)
        start_pos = torch.tensor([0])
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            ep = torch.export.export(Module(), (query, cache, start_pos))
            args = (query, cache, start_pos)
            self.assertEqual(ep.module()(*args), Module()(*args))
            args = (query, cache, torch.tensor([3]))
            self.assertEqual(ep.module()(*args), Module()(*args))
            args = (query, cache, torch.tensor([126]))
            self.assertEqual(ep.module()(*args), Module()(*args))

    def test_none_input_output(self):
        class Z(torch.nn.Module):
            def forward(self, x, y):
                return x * x

        ep = torch.export.export(Z(), (torch.tensor(3), None))
        res = ep.module()(torch.tensor(4), None)
        self.assertEqual(res, torch.tensor(16))

        class B(torch.nn.Module):
            def forward(self, x, y):
                return x * x, y

        ep = torch.export.export(B(), (torch.tensor(3), None))
        res = ep.module()(torch.tensor(4), None)
        self.assertEqual(res[0], torch.tensor(16))
        self.assertEqual(res[1], None)

        decomp = ep.run_decompositions()
        gm = decomp.module()
        res = gm(torch.tensor(4), None)
        self.assertEqual(res[0], torch.tensor(16))
        self.assertEqual(res[1], None)

    def test_print(self):
        class M(torch.nn.Module):
            def forward(self, x):
                print("start")
                x1 = x + x
                print(x1)
                x2 = x1 * x1
                print(1, 2, 3)
                x3 = x2 + x2
                return (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
        )

    def test_logging_logger(self):
        logger = logging.getLogger(__name__)

        class M(torch.nn.Module):
            def forward(self, x):
                logger.log("start")
                x1 = x + x
                logger.debug(x1)
                x2 = x1 * x1
                logger.info(1, 2, 3)
                x3 = x2 + x2
                return (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
        )

    @unittest.skipIf(not TEST_TRANSFORMERS, "No transformers")
    def test_hf_logging_logger(self):
        import transformers

        logger = transformers.utils.logging.get_logger(__name__)

        class M(torch.nn.Module):
            def forward(self, x):
                logger.warning_once("start")
                x1 = x + x
                x2 = x1 * x1
                x3 = x2 + x2
                return (x1, x3)

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
        )

    def test_warning(self):
        class M(torch.nn.Module):
            def forward(self, x):
                warnings.warn("moo")
                res = x + x
                warnings.warn(f"{res}")
                return res

        gm = export(M(), (torch.randn(3, 3),)).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    return (add,)""",
        )

    def test_constant_fqn(self):
        class Nested(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.constant = torch.rand(2, 3)
                self.parameter = torch.nn.Parameter(torch.rand(2, 3))

            def forward(self, x):
                return x + self.constant

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                return self.nested(x) + self.nested.constant + self.nested.parameter

        m = Mod()
        ep = export(m, (torch.rand(2, 3),), strict=True)
        self.assertEqual(ep.constants["nested.constant"], m.nested.constant)
        self.assertEqual(ep.module()(torch.ones(2, 3)), m(torch.ones(2, 3)))

    def test_constant_name(self):
        class Nested(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.constant = torch.rand(2, 3)
                self.parameter = torch.nn.Parameter(torch.rand(2, 3))

            def forward(self, x):
                return x + self.constant

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested_1 = Nested()
                self.nested_2 = Nested()

            def forward(self, x):
                return (
                    self.nested_1(x)
                    + self.nested_2(x)
                    + self.nested_1.constant
                    + self.nested_2.constant
                    + self.nested_1.parameter
                    + self.nested_2.parameter
                )

        m = Mod()
        ep = export(m, (torch.rand(2, 3),), strict=False)
        self.assertEqual(ep.module()(torch.ones(2, 3)), m(torch.ones(2, 3)))

        # check constant fqn when there are multiple instances of the same class
        self.assertEqual(ep.constants["nested_1.constant"], m.nested_1.constant)
        self.assertEqual(ep.constants["nested_2.constant"], m.nested_2.constant)

        # check constant_name in the graph
        placeholders = [
            node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(len(placeholders), 5)
        self.assertTrue(all(ph.name == ph.target for ph in placeholders))
        # suffix should be added to duplicated constant_name
        self.assertEqual(placeholders[2].name, "c_nested_1_constant")
        self.assertEqual(placeholders[3].name, "c_nested_2_constant")

    def test_nested_retrace(self):
        class Nested(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                return x + self.param

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                return x + self.nested(x)

        # first export
        foo = Foo().to("meta")
        inputs = (torch.ones(3, device="meta"),)
        foo(*inputs)
        ep = torch.export.export(foo, inputs, strict=False)

        # second export
        foo_1 = ep.module()
        ep_1 = torch.export.export(foo_1, inputs, strict=False)

        for node1, node2 in zip(ep.graph.nodes, ep_1.graph.nodes):
            nn_module_stack_1 = node1.meta.get("nn_module_stack", None)
            nn_module_stack_2 = node2.meta.get("nn_module_stack", None)

            if nn_module_stack_1 is None:
                self.assertTrue(nn_module_stack_2 is None)
            else:
                for v1, v2 in zip(
                    nn_module_stack_1.values(), nn_module_stack_2.values()
                ):
                    self.assertEqual(v1, v2)

    def test_duplicated_getitem(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.topk(x, 2)

        foo = Foo()
        inputs = (torch.randn(3),)
        ep = torch.export.export(foo, inputs, strict=False)

        graph_module = copy.deepcopy(ep.graph_module)

        call_function_node = None
        num_getitems = 0
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.topk.default
            ):
                call_function_node = node
            elif node.op == "call_function" and node.target == operator.getitem:
                self.assertIs(node.args[0], call_function_node)
                num_getitems += 1

        self.assertIsNotNone(call_function_node)
        self.assertEqual(num_getitems, 2)

        output_node = list(graph_module.graph.nodes)[-1]

        nodes = []
        with graph_module.graph.inserting_before(output_node):
            nodes.append(
                graph_module.graph.call_function(
                    operator.getitem, (call_function_node, 1)
                )
            )
            nodes.append(
                graph_module.graph.call_function(
                    operator.getitem, (call_function_node, 0)
                )
            )
            nodes.append(
                graph_module.graph.call_function(
                    operator.getitem, (call_function_node, 0)
                )
            )
            nodes.append(
                graph_module.graph.call_function(
                    operator.getitem, (call_function_node, 1)
                )
            )
        signature = ExportGraphSignature(
            input_specs=ep.graph_signature.input_specs,
            output_specs=ep.graph_signature.output_specs
            + [
                OutputSpec(
                    kind=OutputKind.USER_OUTPUT,
                    arg=TensorArgument(name=node.name),
                    target=None,
                )
                for node in nodes
            ],
        )
        output_node.args = (output_node.args[0] + tuple(nodes),)
        graph_module.recompile()
        new_ep = ep._update(graph_module, signature)

        new_num_getitems = 0
        for node in new_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.topk.default
            ):
                call_function_node = node
            elif node.op == "call_function" and node.target == operator.getitem:
                self.assertIs(node.args[0], call_function_node)
                new_num_getitems += 1
        self.assertEqual(num_getitems, new_num_getitems)
        self.assertEqual(
            len(list(new_ep.graph.nodes)[-1].args[0]), len(signature.output_specs)
        )


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestExportCustomClass(TorchTestCase):
    def setUp(self):
        if IS_FBCODE:
            lib_file_path = "//caffe2/test/cpp/jit:test_custom_class_registrations"
        elif IS_SANDCASTLE or IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
        torch.ops.load_library(str(lib_file_path))

    def test_lift_custom_obj(self):
        # TODO: fix this test once custom class tracing is implemented

        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export(f, inputs)

        # Replace one of the values with an instance of our custom class
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    setattr(ep.graph_module, "custom_obj", custom_obj)
                    getattr_node = ep.graph.get_attr("custom_obj")
                    # Copy over an nn_module_stack as they are required.
                    getattr_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (getattr_node,),
                    )
                    custom_node.meta["val"] = torch.ones(4, 4)
                    # Copy over an nn_module_stack as they are required.
                    custom_node.meta["nn_module_stack"] = node.meta["nn_module_stack"]
                    custom_node.meta["torch_fn"] = (
                        "custom_op",
                        "torch.ops._TorchScriptTesting.take_an_instance.default",
                    )
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        from torch._export.passes.lift_constants_pass import lift_constants_pass
        from torch._export.serde.serialize import deserialize, serialize

        constants = lift_constants_pass(ep.graph_module, ep.graph_signature, {})
        for k, v in constants.items():
            assert k not in ep.constants
            ep._constants[k] = v
        serialized_vals = serialize(ep)
        deserialized_ep = deserialize(serialized_vals)

        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target
                == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                arg = node.args[0]
                self.assertTrue(arg.op == "placeholder")

    def test_preserve_non_cia_op(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.elu(x)

        ep = export(M(), (torch.randn(2, 3, 4, 5),))
        FileCheck().check_count("torch.ops.aten.elu.default", 1, exactly=True).run(
            ep.graph_module.code
        )

        if IS_FBCODE:
            ep = ep.run_decompositions(_preserve_ops=(torch.ops.aten.elu.default,))
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.elu.default]

            ep = ep.run_decompositions(
                decomp_table=decomp_table,
            )
        FileCheck().check_count("torch.ops.aten.elu.default", 1, exactly=True).run(
            ep.graph_module.code
        )

    def test_preserve_cia_op(self):
        class StaticResizeBilinear2dModule(torch.nn.Module):
            def forward(self, x):
                a = torch.nn.functional.interpolate(
                    x,
                    size=(x.shape[2] * 2, x.shape[3] * 3),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                return a

        ep = export(StaticResizeBilinear2dModule(), (torch.randn(2, 3, 4, 5),))
        FileCheck().check_count(
            "torch.ops.aten.upsample_bilinear2d.vec", 1, exactly=True
        ).run(ep.graph_module.code)

        if IS_FBCODE:
            ep = ep.run_decompositions(
                _preserve_ops=(torch.ops.aten.upsample_bilinear2d.vec,)
            )
        else:
            decomp_table = default_decompositions()
            del decomp_table[torch.ops.aten.upsample_bilinear2d.vec]
            ep = ep.run_decompositions(
                decomp_table=decomp_table,
            )

        FileCheck().check_count(
            "torch.ops.aten.upsample_bilinear2d.vec", 1, exactly=True
        ).run(ep.graph_module.code)


if __name__ == "__main__":
    run_tests()
