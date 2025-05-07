# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# flake8: noqa
import copy
import dataclasses
import logging
import math
import operator
import re
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from re import escape
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
import torch.utils._pytree as pytree
from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch._decomp import decomposition_table
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import normalize_gm
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.scan import scan
from torch._inductor.compile_fx import split_const_gm
from torch._subclasses import FakeTensorMode
from torch.export import (
    default_decompositions,
    Dim,
    export,
    export_for_training,
    unflatten,
)
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
    xfailIfDistributedNotSupported,
)
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
    TEST_WITH_CROSSREF,
    TestCase as TorchTestCase,
)
from torch.testing._internal.custom_tensor import (
    ConstantExtraMetadataTensor,
    CustomTensorPlainOut,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib
from torch.testing._internal.triton_utils import requires_cuda, requires_gpu
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._pytree import (
    LeafSpec,
    register_constant,
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
)


if HAS_GPU:
    import triton
    import triton.language as tl

    from torch._library import capture_triton

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
class Inp1:
    x: Tensor
    y: List[Tensor]
    z: Dict[str, Tensor]


@dataclass
class Inp2:
    a: Tensor
    b: Tensor


@dataclass
class Inp3:
    f: torch.Tensor
    p: torch.Tensor


NON_STRICT_SUFFIX = "_nonstrict"
STRICT_SUFFIX = "_strict"
INLINE_AND_INSTALL_STRICT_SUFFIX = "_inline_and_install_strict"
RETRACEABILITY_STRICT_SUFFIX = "_retraceability_strict"
RETRACEABILITY_NON_STRICT_SUFFIX = "_retraceability_nonstrict"
SERDES_SUFFIX = "serdes"
SERDES_STRICT_SUFFIX = "_serdes_strict"
SERDES_NON_STRICT_SUFFIX = "_serdes_nonstrict"
PREDISPATCH_SUFFIX = "_pre_dispatch"
TRAINING_IR_DECOMP_STRICT_SUFFIX = "_training_ir_to_decomp_strict"
TRAINING_IR_DECOMP_NON_STRICT_SUFFIX = "_training_ir_to_decomp_nonstrict"
LEGACY_EXPORT_STRICT_SUFFIX = "_legacy_export_strict"
LEGACY_EXPORT_NONSTRICT_SUFFIX = "_legacy_export_nonstrict"
CPP_RUNTIME_STRICT_SUFFIX = "_cpp_runtime_strict"
CPP_RUNTIME_NONSTRICT_SUFFIX = "_cpp_runtime_nonstrict"


# Now default mode is non strict, so original unammended test names
# should be treated as non-strict
def is_non_strict_test(test_name):
    return not test_name.endswith(STRICT_SUFFIX)


def is_non_strict_legacy_test(test_name):
    return test_name.endswith(LEGACY_EXPORT_NONSTRICT_SUFFIX)


def is_legacy_test(test_name):
    return test_name.endswith(LEGACY_EXPORT_NONSTRICT_SUFFIX) or test_name.endswith(
        LEGACY_EXPORT_STRICT_SUFFIX
    )


def is_inline_and_install_strict_test(test_name: str) -> bool:
    return test_name.endswith(INLINE_AND_INSTALL_STRICT_SUFFIX)


def is_retracebility_test(test_name):
    return test_name.endswith(RETRACEABILITY_STRICT_SUFFIX) or test_name.endswith(
        RETRACEABILITY_NON_STRICT_SUFFIX
    )


def is_serdes_test(test_name):
    return test_name.endswith(SERDES_STRICT_SUFFIX) or test_name.endswith(
        SERDES_NON_STRICT_SUFFIX
    )


def need_serdes_test(test_name):
    return SERDES_SUFFIX in test_name


def is_training_ir_test(test_name):
    return test_name.endswith(TRAINING_IR_DECOMP_STRICT_SUFFIX) or test_name.endswith(
        TRAINING_IR_DECOMP_NON_STRICT_SUFFIX
    )


def is_cpp_runtime_test(test_name):
    return test_name.endswith(CPP_RUNTIME_STRICT_SUFFIX) or test_name.endswith(
        CPP_RUNTIME_NONSTRICT_SUFFIX
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

    def test_export_slice_unbacked_dim1(self):
        class MySlice(torch.nn.Module):
            def forward(self, x, seq_len):
                l = seq_len.item()
                torch._check_is_size(l, max=x.size(1))
                x = x.narrow(1, 0, l)
                return x

        x = torch.randn(10, 7)
        seq_len = torch.tensor(5)
        torch.export.export(MySlice(), args=(x, seq_len))

    @torch.fx.experimental._config.patch(backed_size_oblivious=True)
    def test_reshape_view_backed_size_oblivious(self):
        N = 3

        class MyModel(torch.nn.Module):
            def forward(self, x):
                y = x[:-1, :]  # [s0 - 1, 32]
                stacked = torch.stack([y] * N, dim=0)  # [N * (s0 - 1), 32]
                reshaped = stacked.reshape(-1, N, 32)  # [(s0 - 1), N, 32]
                return reshaped

        inps = (torch.randn(10, 32),)
        spec = {
            "x": (Dim.AUTO, Dim.STATIC),
        }
        ep = export(MyModel(), inps, dynamic_shapes=spec)

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

    def test_export_strict_narrow_unbacked_expr(self):
        # Tests that we are able to handle 0/1 specialization on sizes represented
        # by unbacked int expressions by transforming them into an unbacked int.
        #
        # This test only works with strict=True, since it relies on dynamo tracing
        # for transforming the expression into an unbacked SymInt.

        def identity(x):
            return x

        class Module(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x, p):
                u0 = p.item()
                torch._check(u0 + 5 <= x.shape[0])
                torch._check(u0 >= 0)
                # Create a tensor of size: (x.shape[0] - u0 - 5).
                return x.narrow(0, u0 + 5, self.fn(x.shape[0] - u0 - 5))

        inputs = (torch.arange(10), torch.tensor(2))

        # Without transforming the unbacked int expression, we can't export.
        with self.assertRaisesRegex(
            RuntimeError, escape("Could not guard on data-dependent expression")
        ):
            export(Module(identity), inputs, strict=True)

        # It works if we transform the whole unbacked int expression into
        # an unbacked int.
        export(Module(torch.sym_fresh_size), inputs, strict=True)


class InputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, y):
        return self.linear(x) * y


class InputModuleWithNestedSubclass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.ones(2, 2))
        self.p2 = torch.nn.Parameter(
            CustomTensorPlainOut(
                CustomTensorPlainOut(
                    torch.Tensor([[0, 0], [0, 1]]),
                    torch.Tensor([[0, 0], [1, 0]]),
                ),
                CustomTensorPlainOut(
                    torch.Tensor([[1, 0], [0, 0]]),
                    torch.Tensor([[0, 1], [0, 0]]),
                ),
            )
        )

    def forward(self, x):
        a = (x + 2 * self.p1 + self.p2).sum().sum()
        return x + a


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

    def test_bincount(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                weights = torch.linspace(0, 1, steps=5)
                bc = x.bincount(weights)
                return bc

        model = M()
        ep = export(model, (torch.randint(0, 8, (5,), dtype=torch.int64),))
        print(ep)
        inp = torch.randint(0, 8, (5,), dtype=torch.int64)
        self.assertTrue(torch.allclose(ep.module()(inp), M()(inp)))

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

    @testing.expectedFailureCppSerDes  # Cpp serder seems to fail parsing complicated guards
    def test_export_statically_known_true(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                shape = y.shape[0] ** 2 - 3 * y.shape[0]
                end = shape
                return x[:, :end]

        dynamic_shapes = (
            (torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
            (torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
        )

        ep = export(
            Foo(),
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        FileCheck().check_count("torch.ops.aten.slice.Tensor", 2, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("operator.sub", 1, exactly=True).run(str(ep.graph))

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

    def test_unbacked_bincount(self):
        class Foo(torch.nn.Module):
            def forward(self, xs):
                u0, u1 = xs.tolist()
                x = torch.ones(u0, dtype=torch.int64)
                y = torch.bincount(x, minlength=u1)
                return y

        m = Foo()
        x = torch.tensor([20, 10])
        ep = export(m, (x,))
        self.assertTrue(torch.allclose(ep.module()(x), m(x)))
        y = torch.tensor([5, 10])
        self.assertTrue(torch.allclose(ep.module()(y), m(y)))

    @requires_gpu
    def test_export_custom_triton_kernel(self):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args=())
        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        class M(torch.nn.Module):
            def forward(self, x, y):
                return custom_add(x, y)

        args = (
            torch.randn(3, device=GPU_TYPE),
            torch.randn(3, device=GPU_TYPE),
        )
        max_len = 128
        dynamic_shapes = {
            "x": {0: Dim("dim0_x", max=max_len)},
            "y": {0: Dim("dim0_y", max=max_len)},
        }
        m = M()
        ep = export(m, args, dynamic_shapes=dynamic_shapes)

        FileCheck().check_count("torch.ops.mylib.add", 1, exactly=True).run(
            ep.graph_module.code
        )
        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=False)
        FileCheck().check_count("torch.ops.mylib.add", 1, exactly=True).run(
            ep.graph_module.code
        )
        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=True)
        FileCheck().check_count(
            "torch.ops.higher_order.triton_kernel_wrapper_functional", 1, exactly=True
        ).run(ep_decomposed.graph_module.code)
        exp_out = m(*args)
        self.assertEqual(exp_out, ep.module()(*args))

    @requires_gpu
    @testing.expectedFailureLegacyExportNonStrict  # Old export graph contains auto_functionalize not Triton wrapper
    @testing.expectedFailureLegacyExportStrict  # Old export graph contains auto_functionalize not Triton wrapper
    def test_export_custom_triton_kernel_mutable(self):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args={"output"})
        def custom_add_out(
            x: torch.Tensor, y: torch.Tensor, output: torch.Tensor
        ) -> torch.Tensor:
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output.clone()

        class M(torch.nn.Module):
            def forward(self, x, y, out):
                return custom_add_out(x, y, out)

        args = (
            torch.randn(3, device=GPU_TYPE),
            torch.randn(3, device=GPU_TYPE),
            torch.zeros(3, device=GPU_TYPE),
        )
        custom_add_out(*args)
        max_len = 128
        dynamic_shapes = {
            "x": {0: Dim("dim0_x", max=max_len)},
            "y": {0: Dim("dim0_y", max=max_len)},
            "out": {0: Dim("dim0_z", max=max_len)},
        }

        m = M()
        ep = export(m, args, dynamic_shapes=dynamic_shapes)

        FileCheck().check_count("torch.ops.mylib.add", 1, exactly=True).run(
            ep.graph_module.code
        )
        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=False)
        FileCheck().check_count(
            "torch.ops.higher_order.auto_functionalized", 1, exactly=True
        ).run(ep_decomposed.graph_module.code)

        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=True)
        if is_training_ir_test(self._testMethodName):
            # TODO: For training IR test, we functionalize the custom triton op with auto_functionalized.
            # The custom op's functional decomposition is not triggered as a result. It might be better to
            # decompose the custom triton ops. Users can workaround by unwrapping auto_functionalized
            # in order to get the functional triton hop if needed.
            FileCheck().check_count(
                "torch.ops.higher_order.auto_functionalized", 1, exactly=True
            ).run(ep_decomposed.graph_module.code)
        else:
            FileCheck().check_count(
                "torch.ops.higher_order.triton_kernel_wrapper_functional",
                1,
                exactly=True,
            ).run(ep_decomposed.graph_module.code)

        x, y, out = (
            torch.randn(3, device=GPU_TYPE),
            torch.randn(3, device=GPU_TYPE),
            torch.zeros(3, device=GPU_TYPE),
        )
        exp_out = m(x, y, out)
        out_copy = out.clone()
        out_copy2 = out.clone()
        out_copy3 = out.clone()
        self.assertEqual(exp_out, ep.module()(x, y, out_copy))
        # For non-functional graph module, out_copy is mutated
        self.assertEqual(out, out_copy)
        self.assertEqual(exp_out, ep_decomposed.module()(x, y, out_copy2))
        # For non-functional graph module, out_copy is not mutated
        self.assertEqual(out_copy2, out_copy3)

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

    def test_nonzero_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, as_tuple: bool) -> torch.Tensor:
                return torch.nonzero(x, as_tuple=as_tuple)

        # Case 1 and 2: as_tuple is True and as_tuple is False.
        for as_tuple in [True, False]:
            example_args = (torch.randn(3, 4, 5), as_tuple)
            dim0_x_max, dim1_x_max = 100, 7
            dynamic_shapes = {
                "x": {
                    0: Dim("dim0_x", max=dim0_x_max),
                    1: Dim("dim1_x_max", max=dim1_x_max),
                },
                "as_tuple": None,
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

        # Case 3: Test special case when input has zero dimensions and a nonzero
        # scalar value.
        example_args = (torch.tensor(10), as_tuple)
        dim0_x_max = 100
        dynamic_shapes = {
            "x": None,
            "as_tuple": None,
        }
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        # Test that the expected upper bound is equal to 1, since our output
        # for this edge case should always be a tensor of size 1.
        vr_upper_bounds = [
            vr.upper for vr in exported_program.range_constraints.values()
        ]
        for vr_upper in vr_upper_bounds:
            self.assertEqual(vr_upper, 1)

    def test_mask_nonzero_static(self):
        class TestModule(torch.nn.Module):
            def forward(self, seq_embeddings, mask, exp):
                # Instead of `output = seq_embeddings[mask]`` which makes
                # output.shape have unbacked symint, encode side knowledge of
                # output.shape as exp.shape to force it to have backed symint
                index = torch.nonzero_static(mask, size=exp.shape[0])
                chunked_index = index.chunk(chunks=mask.dim(), dim=1)
                output = seq_embeddings[chunked_index].squeeze()
                final_output = output * 2
                return final_output

        m = TestModule()

        seq_embeddings = torch.randn(5, 5)
        mask = torch.ones(5, 5, dtype=torch.bool)
        exp = torch.randn(25)
        output = m(seq_embeddings, mask, exp)

        batch = torch.export.Dim("batch")
        exp_size = torch.export.Dim("exp_size", max=100)
        ep = export(
            m,
            (seq_embeddings, mask, exp),
            dynamic_shapes={
                "seq_embeddings": (batch, None),
                "mask": (batch, None),
                "exp": (exp_size,),
            },
        )
        ep_output = ep.module()(seq_embeddings, mask, exp)
        self.assertTrue(torch.allclose(output, ep_output))

        seq_embeddings = torch.randn(6, 5)
        mask = torch.ones(6, 5, dtype=torch.bool)
        exp = torch.randn(30)
        output = m(seq_embeddings, mask, exp)
        ep_output = ep.module()(seq_embeddings, mask, exp)
        self.assertTrue(torch.allclose(output, ep_output))

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
        with self.assertRaisesRegex(RuntimeError, "to be equal to 6, but got 5"):
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

    def test_symint_item(self):
        class M(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        input = (torch.tensor([1], dtype=torch.int),)

        orig_res = M()(*input)
        ep_res = torch.export.export(M(), input).module()(*input)
        self.assertEqual(orig_res, ep_res)

    def test_symbool_item(self):
        class M(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        input = (torch.tensor([1], dtype=torch.bool),)

        orig_res = M()(*input)
        ep_res = torch.export.export(M(), input).module()(*input)
        self.assertEqual(orig_res, ep_res)

    def test_symfloat_item(self):
        class M(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        input = (torch.tensor([3.14], dtype=torch.float),)

        orig_res = M()(*input)
        ep_res = torch.export.export(M(), input).module()(*input)
        self.assertEqual(orig_res, ep_res)

    def test_unbacked_to_cond(self):
        strict = True

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
        torch.export.export(M(), (torch.randn(7),), strict=strict)

    def test_unbacked_to_cond_passthrough(self):
        strict = True

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
        torch.export.export(M(), (torch.randn(7),), strict=strict)

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

        ep = export(M(), (torch.randn(2, 3),), strict=False).run_decompositions({})
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

        ep = export(M(), (torch.randn(2, 3),), strict=False).run_decompositions({})
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

    def test_malformed_fqn_from_source_name(self):
        # See https://github.com/pytorch/pytorch/issues/141939
        from types import MethodType

        class Block(torch.nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.to_out = torch.nn.ModuleList([])
                self.to_out.append(torch.nn.Linear(i, o, bias=True))
                self.to_out.append(torch.nn.Dropout(0.5))

            def forward(self, x):
                for l in self.to_out:
                    x = l(x)
                return x

        class Problem1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleDict(
                    {f"{i}": Block(64, 64) for i in range(5)}
                )

            def forward(self, x):
                for k, m in self.blocks.items():
                    x = m(x)
                return x

        class Problem2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(64, 64) for i in range(5)])

            def forward(self, x):
                x = self.blocks[0](x)
                for m in self.blocks[1:4]:
                    x = m(x)
                return x

        def _split_after_forward(self, *args, **kwargs):
            return self._orig_forward(*args, **kwargs)

        def annotate_split_points(mod: torch.nn.Module, spec):
            for qualname, split_type in spec.items():
                atoms = qualname.split(".")
                predecessor_module = mod
                for i, atom in enumerate(atoms[:-1]):
                    try:
                        predecessor_module = getattr(predecessor_module, atom)
                    except AttributeError as e:
                        raise e
                mod_to_wrap = getattr(predecessor_module, atoms[-1])
                mod_to_wrap._orig_forward = mod_to_wrap.forward
                mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)

        for problem in [Problem1, Problem2]:
            m = problem()
            m(torch.rand(64, 64))
            # simpified torch.distributed.pipeline code
            annotate_split_points(m, {"blocks.1": 1, "blocks.3": 1})
            gm = export(m, (torch.rand(64, 64),))
            torch.export.unflatten(gm)

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

    def test_output_node_name(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = TestModule()
        x = torch.randn(20, 10)
        ep_model = export(model, (x,), strict=False).module()
        self.assertEqual(list(ep_model.graph.nodes)[-1].name, "output")
        self.assertTrue(torch.allclose(model(x), ep_model(x)))

    def test_real_tensor_size_mismatch(self):
        from torch._subclasses.fake_tensor import MetadataMismatchError

        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.mylib.foo(a, b)

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        @foo.register_fake
        def foo_fake_impl(a, b):
            m, n = a.shape
            return torch.empty(n, m)  # incorrectly permute

        error_type = (
            MetadataMismatchError
            if is_non_strict_test(self._testMethodName)
            else torch._dynamo.exc.TorchRuntimeError
        )
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            # won't catch anything if dims are equal
            export(
                M(),
                (torch.randn(4, 4), torch.randn(4, 4)),
            )
            # catch concrete inequality
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found an output size mismatch between fake shape 8 and real shape 4, "
                r"at output\.size\(0\), for func: mylib.foo.default",
            ):
                export(
                    M(),
                    (torch.randn(4, 8), torch.randn(4, 8)),
                )
            # same test with dynamic shapes
            d0 = Dim("d0")
            d1 = Dim("d1")
            export(
                M(),
                (torch.randn(4, 4), torch.randn(4, 4)),
                dynamic_shapes={
                    "a": (d0, d1),
                    "b": (d0, d1),
                },
            )
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found an output size mismatch between fake shape s\d+ and real shape 4, "
                r"at output\.size\(0\), for func: mylib.foo.default",
            ):
                export(
                    M(),
                    (torch.randn(4, 8), torch.randn(4, 8)),
                    dynamic_shapes={
                        "a": (d0, d1),
                        "b": (d0, d1),
                    },
                )

    def test_real_tensor_alias_dtype_mismatch(self):
        from torch._subclasses.fake_tensor import MetadataMismatchError

        error_type = (
            MetadataMismatchError
            if is_non_strict_test(self._testMethodName)
            else torch._dynamo.exc.TorchRuntimeError
        )

        # test alias case
        class M(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo_alias(a)

        @torch.library.custom_op("mylib::foo_alias", mutates_args={})
        def foo_alias(a: torch.Tensor) -> torch.Tensor:
            return a * 2

        @foo_alias.register_fake
        def foo_fake_impl(a):
            return a

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found an aliasing mismatch between fake output (.*\n)*.* "
                r"and real output (.*\n)*.* for func: mylib.foo_alias.default",
            ):
                ep = export(M(), (torch.randn(4, 4),))

        # test dtype case
        class N(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo_dtype(a)

        @torch.library.custom_op("mylib::foo_dtype", mutates_args={})
        def foo_dtype(a: torch.Tensor) -> torch.Tensor:
            return a * 2

        @foo_dtype.register_fake
        def foo_fake_impl(a):
            m, n = a.shape
            return torch.empty([m, n], dtype=torch.int32)

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found a metadata mismatch between fake tensor (.*\n)*.* "
                r"and real tensor (.*\n)*.* at output, for func: mylib.foo_dtype.default",
            ):
                ep = export(N(), (torch.randn(4, 4),))

    def test_real_tensor_for_max_op(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x = x[x > 0]
                y = y[y > 0]
                return max(x.shape[0], y.shape[0])

        model = Foo()
        inputs = (torch.zeros(64), torch.ones(64))
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

        self.assertEqual(ep.module()(*inputs), model(*inputs))
        x = torch.zeros(64)
        y = torch.ones(64)
        # This seems to be a bug with old export because when we pass in x, x
        # as input, runtime assertion should fail. This is because we would create
        # guard on y.shape[0] > x.shape[0] but somehow in old export, we dce this
        # assertion.
        self.assertEqual(ep.module()(x, x), model(x, x))
        self.assertEqual(ep.module()(x, y), model(x, y))

    def test_draft_export_checks_mutation_with_nan(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        @foo.register_fake
        def _(x, y):
            return x + y

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.full((64,), torch.nan), torch.full((64,), torch.nan))
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_checks_mutation(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(x, y):
            return x.clone()

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.randn(64), torch.randn(64))
        with self.assertRaisesRegex(RuntimeError, "for argument 'y'"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={"y"})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(x, y):
            return x.clone()

        # No errors
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_checks_mutation_list(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(xs: List[torch.Tensor]) -> torch.Tensor:
            x, y = xs
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(xs):
            x, y = xs
            return x.clone()

        class Foo(torch.nn.Module):
            def forward(self, xs):
                return foo(xs)

        model = Foo()
        inputs = ([torch.randn(64), torch.randn(64)],)
        with self.assertRaisesRegex(RuntimeError, "for argument 'xs'"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={"xs"})
        def foo(xs: List[torch.Tensor]) -> torch.Tensor:
            x, y = xs
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(xs):
            x, y = xs
            return x.clone()

        # No errors
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_checks_aliasing(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x

        @foo.register_fake
        def _(x, y):
            return x.clone()

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.randn(64), torch.randn(64))
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @foo.register_fake
        def _(x, y):
            return x.clone()

        # No errors
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_infers_fake_kernel(self):
        strict = True
        with torch.library._scoped_library("export", "FRAGMENT") as lib:
            lib.define("bar(Tensor x) -> Tensor")
            lib.impl("bar", lambda x: x[0].clone(), "CPU")

            @torch.library.custom_op("export::foo", mutates_args={})
            def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x * y

            class Foo(torch.nn.Module):
                def forward(self, x, y):
                    return foo(x, y), torch.ops.export.bar(y)

            model = Foo()
            inputs = (torch.randn(1, 3), torch.randn(2, 1))
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs, strict=strict)

        # expecttest only works for the base TestExport class.
        if self.__class__ != TestExport:
            return

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, x, y):
    foo = torch.ops.export.foo.default(x, y);  x = None
    sym_size_int = torch.ops.aten.sym_size.int(foo, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(foo, 1)
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int);  sym_constrain_range_for_size_default = None
    ge = sym_size_int >= 0;  sym_size_int = None
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
    sym_constrain_range_for_size_default_1 = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int_1);  sym_constrain_range_for_size_default_1 = None
    ge_1 = sym_size_int_1 >= 0;  sym_size_int_1 = None
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
    bar = torch.ops.export.bar.default(y);  y = None
    sym_size_int_2 = torch.ops.aten.sym_size.int(bar, 0)
    sym_constrain_range_for_size_default_2 = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int_2);  sym_constrain_range_for_size_default_2 = None
    ge_2 = sym_size_int_2 >= 0;  sym_size_int_2 = None
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_default_2 = None
    return (foo, bar)""",
        )

    def test_draft_export_fake_kernel_inference_errors(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.expand(32, 3).contiguous()[4]

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.randn(1, 3), torch.randn(2, 1))

        with self.assertRaisesRegex(RuntimeError, "non-zero storage offset"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.randn(3, 3).diagonal()

        with self.assertRaisesRegex(RuntimeError, "not dense in memory"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclasses_parameterization(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(torch.ones(3, 4), torch.ones(3, 4))
                )

            def forward(self, x):
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)

        ep_training = torch.export.export_for_training(m, (ref_x,))
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_1), kwargs = {})
    return (add_1,)""",
        )

        ep = export(m, (ref_x,)).run_decompositions({})
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_parametrizations_p2_original0 : [num_users=1] = placeholder[target=p_parametrizations_p2_original0]
    %p_parametrizations_p2_original1 : [num_users=1] = placeholder[target=p_parametrizations_p2_original1]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_parametrizations_p2_original0), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_parametrizations_p2_original1), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %add_1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_2,), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_1), kwargs = {})
    return (add_3,)""",
        )
        res = ep.module()(ref_x)

        self.assertEqual(res, ref_out)

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclasses_parameterization_nested(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(2, 2))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        CustomTensorPlainOut(
                            torch.Tensor([[0, 0], [0, 1]]),
                            torch.Tensor([[0, 0], [1, 0]]),
                        ),
                        CustomTensorPlainOut(
                            torch.Tensor([[1, 0], [0, 0]]),
                            torch.Tensor([[0, 1], [0, 0]]),
                        ),
                    )
                )

            def forward(self, x):
                a = (x + 2 * self.p1 + self.p2).sum().sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(2, 2)
        ref_out = m(ref_x)

        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=2] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_p2), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_1,), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sum_1,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_2), kwargs = {})
    return (add_2,)""",
        )

        ep = export(m, (ref_x,))
        ep = ep.run_decompositions({})
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_parametrizations_p2_original0 : [num_users=1] = placeholder[target=p_parametrizations_p2_original0]
    %p_parametrizations_p2_original1 : [num_users=1] = placeholder[target=p_parametrizations_p2_original1]
    %p_parametrizations_p2_original2 : [num_users=1] = placeholder[target=p_parametrizations_p2_original2]
    %p_parametrizations_p2_original3 : [num_users=1] = placeholder[target=p_parametrizations_p2_original3]
    %x : [num_users=2] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original0), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original1), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %add_2), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original2), kwargs = {})
    %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original3), kwargs = {})
    %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %add_5), kwargs = {})
    %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %add_6), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_7,), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sum_1,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_2), kwargs = {})
    return (add_8,)""",
        )
        res = ep.module()(ref_x)
        self.assertEqual(res, ref_out)

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )
                self.b1 = torch.nn.Buffer(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = (2 * self.p1 + self.p2 + self.b1).sum()
                return x + res.get_elem_a().b

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertTrue(torch.allclose(ep_training.module()(ref_x), ref_out))
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %b_b1 : [num_users=1] = placeholder[target=b_b1]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %b_b1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_1,), kwargs = {})
    %access_subclass_inner_tensor_default_64 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%sum_1, a), kwargs = {})
    %access_subclass_inner_tensor_default_69 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_64, b), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_69), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access_submodule(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )
                self.b1 = torch.nn.Buffer(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                return x

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                res = (2 * self.bar.p1 + self.bar.p2 + self.bar.b1).sum()
                return x + res.get_elem_a().b

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_bar_p1 : [num_users=1] = placeholder[target=p_bar_p1]
    %p_bar_p2 : [num_users=1] = placeholder[target=p_bar_p2]
    %b_bar_b1 : [num_users=1] = placeholder[target=b_bar_b1]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_bar_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_bar_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %b_bar_b1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_1,), kwargs = {})
    %access_subclass_inner_tensor_default_64 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%sum_1, a), kwargs = {})
    %access_subclass_inner_tensor_default_69 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_64, b), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_69), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access_const_metadata(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    ConstantExtraMetadataTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = 2 * self.p1 + self.p2
                res2 = res + res.constant_attribute
                return x + res2.elem.elem

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 4), kwargs = {})
    %access_subclass_inner_tensor_default_10 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_1, elem), kwargs = {})
    %access_subclass_inner_tensor_default_13 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_10, elem), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_13), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access_const_metadata_not_top_level(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    ConstantExtraMetadataTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = 2 * self.p1 + self.p2
                res2 = res + res.constant_attribute
                return x + res2.elem.elem

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 4), kwargs = {})
    %getattr_22 : [num_users=1] = call_function[target=builtins.getattr](args = (%add_1, elem), kwargs = {})
    %getattr_27 : [num_users=1] = call_function[target=builtins.getattr](args = (%getattr_22, elem), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %getattr_27), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access_const_metadata_not_top_level(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = 2 * self.p1 + self.p2
                res2 = res + res.a.elem + res.b.constant_attribute
                return x + res2.a.elem

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %access_subclass_inner_tensor_default_18 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add, a), kwargs = {})
    %access_subclass_inner_tensor_default_21 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_18, elem), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %access_subclass_inner_tensor_default_21), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 4), kwargs = {})
    %access_subclass_inner_tensor_default_25 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_2, a), kwargs = {})
    %access_subclass_inner_tensor_default_28 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_25, elem), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_28), kwargs = {})
    return (add_3,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access_complicated_metadata(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    ConstantExtraMetadataTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = x + 2 * self.p1 + self.p2
                return res.elem.elem + self.p2.get_complicated_metadata().foo

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export_for_training(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_p2), kwargs = {})
    %access_subclass_inner_tensor_default_10 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_1, elem), kwargs = {})
    %access_subclass_inner_tensor_default_13 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_10, elem), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%access_subclass_inner_tensor_default_13, 4), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_real_tensor_errors_on_aliasing_custom_op(self):
        @torch.library.custom_op("export::foo_alias", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x

        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.ops.export.foo_alias(x) * 2

        model = Foo()
        inputs = (torch.randn(4, 4),)
        error_type = (
            RuntimeError
            if is_non_strict_test(self._testMethodName)
            else torch._dynamo.exc.TorchRuntimeError
        )
        with self.assertRaisesRegex(
            error_type,
            (
                r"The output of this custom operator \(1\) must not also be an input "
                r"to this custom operator and \(2\) may not alias any inputs to this "
                r"custom operator or other returns"
            ),
        ):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

    def test_real_tensor_bool_cast(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return bool(x.eq(0.1).any())

        model = Foo()
        inputs = (torch.randn(64),)
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs, strict=False)

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

    @testing.expectedFailureLegacyExportNonStrict  # Trivial error, just need to move the error check earlier, for real users it wont matter
    @testing.expectedFailureLegacyExportStrict  # Trivial error, just need to move the error check earlier, for real users it wont matter
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
            r"You marked.*but your code specialized it to be a constant.*less strict API such as maybe_mark_dynamic or Dim.AUTO.",
        ):
            export(Foo(), inputs, dynamic_shapes=shapes)

    def test_dim_dynamic_specialization(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + 2

        # 0/1 specialization
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified dim hint Dim.DYNAMIC.*"
            r"but export 0/1 specialized due to hint of 0 for dimension "
            r"inputs\['x'\]\.shape\[0\](.*\n)*.*"
            r"Received user-specified dim hint Dim.DYNAMIC.*"
            r"but export 0/1 specialized due to hint of 1 for dimension "
            r"inputs\['x'\]\.shape\[1\].*",
        ):
            export(
                Foo(),
                (torch.randn(0, 1),),
                dynamic_shapes={
                    "x": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                },
            )

        class Bar(torch.nn.Module):
            def forward(self, x):
                assert x.shape[0] <= 32
                return x + 2

        # static specialization
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified dim hint Dim.DYNAMIC.*"
            r"but tracing inferred a static shape of 32 for dimension "
            r"inputs\['x'\]\.shape\[0\](.*\n)*.*",
        ):
            export(
                Bar(),
                (torch.randn(32),),
                dynamic_shapes={
                    "x": {0: Dim.DYNAMIC(min=32)},
                },
            )

    def test_dim_hint_ranges(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = (
            torch.randn(6, 4),
            torch.randn(6, 4),
        )
        shapes = {
            "x": (Dim.AUTO(min=4), Dim.AUTO),
            "y": (Dim.DYNAMIC(max=16), Dim.AUTO(max=32)),
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        ep.module()(torch.randn(8, 5), torch.randn(8, 5))
        with self.assertRaisesRegex(
            RuntimeError, "Expected input at .* to be >= 4, but got 3"
        ):
            ep.module()(torch.randn(3, 5), torch.randn(3, 5))
        with self.assertRaisesRegex(
            RuntimeError, "Expected input at .* to be <= 16, but got 17"
        ):
            ep.module()(torch.randn(17, 5), torch.randn(17, 5))
        with self.assertRaisesRegex(
            RuntimeError, "Expected input at .* to be <= 32, but got 33"
        ):
            ep.module()(torch.randn(9, 33), torch.randn(9, 33))

    def test_dim_hint_range_violations(self):
        class Foo(torch.nn.Module):
            def forward(self, xs):
                x, y = xs["data"][0]
                assert y.shape[0] <= 32
                return x[6:], y + 2

        x, y = torch.randn(8), torch.randn(8)

        # conflict with lower bound
        shapes = torch.export.ShapesCollection()
        shapes[x] = [Dim.DYNAMIC(max=5)]
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[None, 5\], conflicting with the inferred .*"
            r"\[6, int_oo\],.* for inputs\['xs'\]\['data'\]\[0\]\[0\]\.shape\[0\]",
        ):
            export(Foo(), ({"data": [[x, y]]},), dynamic_shapes=shapes)

        # conflict with upper bound
        shapes = torch.export.ShapesCollection()
        shapes[y] = [Dim.AUTO(min=48, max=62)]
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[48, 62\], conflicting with the inferred .*"
            r"\[2, 32\],.* for inputs\['xs'\]\['data'\]\[0\]\[1\]\.shape\[0\]",
        ):
            export(Foo(), ({"data": [[x, y]]},), dynamic_shapes=shapes)

        class Bar(torch.nn.Module):
            def forward(self, x):
                return x + 2

        # conflict with static range
        shapes = {"x": [Dim.STATIC(min=6, max=8)]}
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[6, 8\], conflicting with the inferred .*"
            r"\[4, 4\],.* for inputs\['x'\].shape\[0\]",
        ):
            export(Bar(), (torch.randn(4),), dynamic_shapes=shapes)

        # multiple conflicts
        class Moo(torch.nn.Module):
            def forward(self, x, y):
                assert x.shape[0] <= 32
                assert y.shape[0] >= 128
                return x + 2, y + 2

        inps = (torch.randn(16), torch.randn(256))
        shapes = {
            "x": (Dim.DYNAMIC(min=33),),
            "y": (Dim.DYNAMIC(max=127),),
        }
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[33, None\], conflicting with the inferred .*"
            r"\[2, 32\],.* for inputs\['x'\].shape\[0\](.*\n)*.*"
            r"Received user-specified .* \[None, 127\], conflicting with the inferred .*"
            r"\[128, int_oo\],.* for inputs\['y'\].shape\[0\]",
        ):
            export(Moo(), inps, dynamic_shapes=shapes)

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

    def test_export_custom_op_lib(self):
        ops_registered_before = set(torch.ops.mylib)

        # Assert warning for CompositeImplictAutograd op
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

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
        decomp_table = default_decompositions()
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
    split_with_sizes = torch.ops.aten.split_with_sizes.default(linear, [1, 1, 1]);  linear = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
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

    def test_is_exporting(self):
        class Mod(torch.nn.Module):
            def forward(self, pred, x):
                def f(x):
                    return x.sin() if torch.compiler.is_exporting() else x.cos()

                y = f(x)

                def true_fn(x):
                    return f(x) - 1 if torch.compiler.is_exporting() else f(x) + 1

                def false_fn(x):
                    return f(x) + 1 if torch.compiler.is_exporting() else f(x) - 1

                return torch.cond(pred, true_fn, false_fn, (x,)) * y

        ep = export(
            Mod(),
            (
                torch.tensor(False),
                torch.randn(3, 4),
            ),
        )
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.higher_order.cond", 1, exactly=True).run(
            ep.graph_module.code
        )

        # True graph should contain sin and sub
        FileCheck().check_count("torch.ops.aten.sub", 1, exactly=True).run(
            ep.graph_module.true_graph_0.code
        )
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            ep.graph_module.true_graph_0.code
        )

        # False graph should contain sin and add
        FileCheck().check_count("torch.ops.aten.add", 1, exactly=True).run(
            ep.graph_module.false_graph_0.code
        )
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            ep.graph_module.false_graph_0.code
        )

    def test_ends_of_bounds_oblivious(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(10))

            def forward(self, x, y):
                self.buf[0 : x.shape[0]] = x
                return x + 2, y[:, ::1]

        inps = (torch.randn(10), torch.randn(32, 36))
        dynamic_shapes = {
            "x": {0: Dim("dx", min=1, max=10)},
            "y": {0: Dim("dy0"), 1: Dim("dy1")},
        }
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = export(Foo(), inps, dynamic_shapes=dynamic_shapes)
        ep.module()(torch.randn(9), torch.randn(4, 4))
        ep.module()(torch.randn(1), torch.randn(1, 1))

    def test_colin_unbacked_backed_vr_sub(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                nz = torch.nonzero(a)
                ones = a.new_ones([nz.size(0), b.size(0)])
                torch._check(ones.size(0) >= 1)
                equals = torch.add(ones, c)
                return equals

        model = Model()
        example_inputs = (
            torch.ones(64),
            torch.randn(32),
            torch.randn(64, 32),
        )
        dynamic_shapes = {"a": None, "b": None, "c": (Dim.DYNAMIC, Dim.STATIC)}
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes)

        # check lower bound
        for sym, vr in ep.range_constraints.items():
            if str(sym) in ["u0", "s0"]:
                self.assertEqual(vr.lower, 1)

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
            "of the form 2\\*s92, where s92 is an integer",
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

        # Since symbol names are based on hash of source names, and these differ across inference and
        # training, we do range comparisons instead.
        self.assertEqual(
            str(ep_for_training.range_constraints.values()),
            str(ep_for_real.range_constraints.values()),
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

    def test_range_constraints_with_replacement(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return (x + y)[:3]

        m = M()
        inp = (torch.randn(4), torch.randn(4))
        dynamic_shapes = ((torch.export.Dim.DYNAMIC,), (torch.export.Dim.DYNAMIC,))
        ep = export(m, inp, dynamic_shapes=dynamic_shapes)
        assert len(ep.range_constraints) == 1
        vr = next(iter(ep.range_constraints.values()))
        self.assertEqual(vr.lower, 3)

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
        # specify shape of tensor
        shapes_collection[x] = (dim,)
        # tensor can be arbitrarily deep
        shapes_collection[y[0]] = (dim,)
        # can also specify some dimension in shape of tensor
        shapes_collection[z["k"]][0] = dim

        ep = export(m, args, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

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

    def test_dynamic_shapes_builder_pytree(self):
        torch.export.register_dataclass(
            Inp1,
            serialized_type_name="test_dynamic_shapes_builder_pytree.Inp1",
        )

        class M(torch.nn.Module):
            def forward(self, inp: Inp1):
                return inp.x + inp.y[0] + inp.z["k"]

        m = M()
        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (Inp1(x, y, z),)

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

    def test_dynamic_shapes_inferred_basic(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                # x and y[0] must have same dynamic shape (say `dim`) >= 3
                tmp = (x + y[0])[:3]
                # z["k"] must have static shape = 3
                return tmp * z["k"]

        m = M()
        args = (torch.randn(4), [torch.randn(4)], {"k": torch.randn(3)})

        additional_inputs = torch.export.AdditionalInputs()
        # 4->5, 4->5, 3->3
        good_args = (torch.randn(5), [torch.randn(5)], {"k": torch.randn(3)})
        additional_inputs.add(good_args)

        ep = export(m, args, dynamic_shapes=additional_inputs)
        got_shapes = [
            str(tuple(node.meta["val"].shape))
            for node in ep.graph.find_nodes(op="placeholder")
        ]
        dim = next(iter(ep.range_constraints.keys()))
        expected_shapes = [f"({dim},)", f"({dim},)", "(3,)"]
        self.assertEqual(got_shapes, expected_shapes)

        def expect_error(bad_args, run_time_msg, compile_time_msg):
            with self.assertRaisesRegex(RuntimeError, run_time_msg):
                ep.module()(*bad_args)

            additional_inputs = torch.export.AdditionalInputs()
            additional_inputs.add(bad_args)

            with self.assertRaisesRegex(RuntimeError, compile_time_msg):
                export(m, args, dynamic_shapes=additional_inputs)

        expect_error(
            # 4->2, 4->2, 3->3
            bad_args=(torch.randn(2), [torch.randn(2)], {"k": torch.randn(3)}),
            run_time_msg="Expected input.*to be >= 3, but got 2",
            compile_time_msg="Expected input.*to be >= 3, but got 2",
        )

        expect_error(
            # 4->6, 4->7, 3->3
            bad_args=(torch.randn(6), [torch.randn(7)], {"k": torch.randn(3)}),
            run_time_msg="Expected input.*to be equal to 6, but got 7",
            compile_time_msg="Expected input.*to be equal to 6, but got 7",
        )

        expect_error(
            # 4->5, 4->5, 3->4
            bad_args=(torch.randn(5), [torch.randn(5)], {"k": torch.randn(4)}),
            run_time_msg="Expected input.*to be equal to 3, but got 4",
            compile_time_msg=r"You marked.*but your code specialized it to be a constant.*less strict API such as maybe_mark_dynamic or Dim.AUTO.",
        )

    def test_additional_inputs_constants(self):
        @dataclass
        class D:
            b: bool
            i: int
            f: float
            t: torch.Tensor

        pytree.register_dataclass(D)

        class M(torch.nn.Module):
            def forward(self, d: D):
                return d.i + d.f + d.t

        input1 = (D(True, 3, 3.0, torch.ones(3)),)

        # int and tensor change
        input2 = (D(True, 4, 3.0, torch.ones(4)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        dynamic_shapes = ai.dynamic_shapes(M(), input1)
        self.assertEqual(
            dynamic_shapes, {"d": [None, Dim.DYNAMIC, None, (Dim.DYNAMIC,)]}
        )
        torch.export.export(M(), input1, dynamic_shapes=ai)

        # float changes, error
        input2 = (D(True, 3, 4.0, torch.ones(3)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(3\.0, 3\.0, 4\.0\)"
        ):
            ai.dynamic_shapes(M(), input1)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(3\.0, 3\.0, 4\.0\)"
        ):
            torch.export.export(M(), input1, dynamic_shapes=ai)

        # bool changes, error
        input2 = (D(False, 3, 3.0, torch.ones(3)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(True, True, False\)"
        ):
            ai.dynamic_shapes(M(), input1)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(True, True, False\)"
        ):
            torch.export.export(M(), input1, dynamic_shapes=ai)

        # Differing types
        input1 = (D(True, 0, 3.0, torch.ones(3)),)
        input2 = (D(True, False, 3.0, torch.ones(3)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        with self.assertRaisesRegex(
            ValueError,
            r"differing types, so they cannot be marked as dynamic: \(0, 0, False\)",
        ):
            print(ai.dynamic_shapes(M(), input1))
        with self.assertRaisesRegex(
            ValueError,
            r"differing types, so they cannot be marked as dynamic: \(0, 0, False\)",
        ):
            torch.export.export(M(), input1, dynamic_shapes=ai)

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
        )  # ValueError: Node type mismatch; expected <class 'list'>, but got .*Dim.*.
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

        class O(torch.nn.Module):
            def forward(self, x):
                return x + 2

        inputs = (torch.randn(4, 8, 6),)
        dynamic_shapes = {"x": (dim, None)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Expected dynamic shape spec .* at `dynamic_shapes\['x'\]` to have the same length "
            r"as the actual tensor shape torch\.Size\(\[4, 8, 6\]\) \(expected 3, but got 2 instead\)",
        ):
            export(O(), inputs, dynamic_shapes=dynamic_shapes)

    def test_unbacked_bindings_for_divisible_u_symint(self):
        from torch._export.utils import _get_shape_env_from_gm
        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.mylib.foo_unbacked(a, b)

        @torch.library.custom_op("mylib::foo_unbacked", mutates_args={})
        def foo_unbacked(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a[b.item()]

        @foo_unbacked.register_fake
        def foo_unbacked_fake_impl(a, b):
            ctx = torch.library.get_ctx()
            u = ctx.new_dynamic_size(min=0, max=len(a) // 10) * 10
            return torch.empty(u, a.shape[1], dtype=a.dtype)

        # check binding path is correct
        ep = export(
            M(),
            (torch.randn(100, 4), torch.tensor(10)),
        )
        foo = [node for node in ep.graph.nodes if node.name == "foo_unbacked"][0]
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

        # collect bound symbols
        bound = set()
        for node in ep.graph.nodes:
            bound.update(node.meta.get("unbacked_bindings", {}))

        # check ShapeEnv counters compared to binding indices
        shape_env = _get_shape_env_from_gm(ep.graph_module)
        next_index = next(shape_env.unbacked_symint_counter)
        for symbol in bound:
            self.assertTrue(symbol_is_type(symbol, SymT.UNBACKED_INT))
            self.assertTrue(
                int(str(symbol)[len(prefix_str[SymT.UNBACKED_INT]) :]) < next_index
            )

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

    def test_replaced_unbacked_bindings(self):
        import sympy

        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                m, n = x.item(), y.item()
                torch._check(m == 4)
                torch._check(n == z.shape[0])
                return m + n + z

        inps = (
            torch.tensor(4),
            torch.tensor(5),
            torch.randn(5),
        )
        dynamic_shapes = {
            "x": None,
            "y": None,
            "z": (Dim("dx", max=16),),
        }
        ep = export(Foo(), inps, dynamic_shapes=dynamic_shapes)
        # values should have no unbacked symbols, bindings should be empty
        for node in ep.graph.nodes:
            symbols = []
            val = node.meta.get("val")
            bindings = node.meta.get("unbacked_bindings")
            self.assertTrue(
                not (
                    isinstance(val, sympy.Symbol)
                    and symbol_is_type(val, SymT.UNBACKED_INT)
                )
            )
            self.assertTrue(bindings is None)

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

    @testing.expectedFailureTrainingIRToRunDecomp
    @testing.expectedFailureTrainingIRToRunDecompNonStrict
    def test_unbacked_pad(self):
        class Foo(torch.nn.Module):
            def forward(self, xs, pad):
                u0, u1, u2 = xs.tolist()
                x = torch.ones(u0, u1, u2)
                pl0, pr0, pl1, pr1 = pad.tolist()
                return torch.nn.functional.pad(x, (pl0, pr0, pl1, pr1))

        x = torch.tensor([64, 64, 64])
        pad = torch.tensor([8, -8, 4, 0])
        m = Foo()
        ep = export(m, (x, pad))
        self.assertEqual(ep.module()(x, pad).shape, m(x, pad).shape)

        # don't guard on negative/positive pad values
        pad2 = torch.tensor([-5, 9, 0, 8])
        self.assertEqual(ep.module()(x, pad2).shape, m(x, pad2).shape)

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
                # Could not guard on data-dependent expression Ne(Mod(u1, u2), 0)
                return r.view(items[0], items[2])

        M = M_v0
        with self.assertRaisesRegex(
            error_type,
            "The following call raised this error(.*\n)+"
            f".*{re.escape('return r.view(items[0], items[2])')}(.*\n)+"
            "To fix the error, insert one of the following checks before this call.*:\n"
            f".*{re.escape('torch._check((items[1] % items[2]) == 0)')}.*\n"
            f".*{re.escape('torch._check((items[1] % items[2]) != 0)')}(.*\n)+"
            f".*{re.escape('(These suggested fixes were derived by replacing `u1` with items[1]')}"
            f".*{re.escape('or r.shape[1], `u2` with items[2] in Eq(Mod(u1, u2), 0) and its negation.')}",
        ):
            export(N(), (t,), strict=strict)

        class M_v1(torch.nn.Module):
            def forward(self, t):
                items = [t[i].item() for i in range(t.numel())]
                r = torch.randn([items[0], items[1]])
                # TODO(pianpwk): this isn't the suggested fixes.
                # fix issue with % being interpreted as PythonMod instead of Mod
                torch._check(items[1] == items[2])
                return r.view(items[0], items[2])

        M = M_v1
        export(N(), (t,), strict=strict)

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
                "torch._check((i // 2) != 0)",
                # Could not guard on data-dependent expression Eq((u0//2), 1)
                "torch._check((i // 2) != 1)",
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

    def test_simple_unbacked_view(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                u0 = x.item()
                y = torch.empty(5, u0)
                return y.view(u0, 5)  # [5, u0] -> [u0, 5]

        ep = export(Foo(), (torch.tensor([9]),))
        self.assertEqual(ep.module()(torch.tensor([8])).size(0), 8)
        self.assertEqual(ep.module()(torch.tensor([5])).size(0), 5)

        class Foov2(torch.nn.Module):
            def forward(self, xs):
                xsl = xs.tolist()
                a, b = xsl
                x = torch.zeros(a)
                return x.reshape(b)

        xs = torch.tensor([4, 4])
        ep = export(Foov2(), (xs,))
        self.assertEqual(ep.module()(xs).size(0), 4)
        self.assertEqual(ep.module()(torch.tensor([5, 5])).size(0), 5)

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
        gm = export(foo, (torch.tensor([2, 3, 5]),)).run_decompositions({})

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

    @testing.expectedFailureCppSerDes  # cpp ser/der not handling complicated symbols
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

        dim = torch.export.Dim.AUTO
        dynamic_shapes = {"x": {2: dim, 3: dim}, "y": {2: dim, 3: dim}}

        exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)
        self.assertEqual(exported_program.module()(*inputs), model(*inputs))

    def test_export_max_onnx_reported(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                s1 = max(x.shape[0], y.shape[0])
                s2 = max(x.shape[1], y.shape[1])
                z = torch.zeros((s1, s2), dtype=x.dtype)
                z[: x.shape[0], : x.shape[1]] = x
                z[: y.shape[0], : y.shape[1]] += y
                return z

        model = Model()
        x = torch.arange(6).reshape((2, 3))
        y = torch.arange(6).reshape((3, 2)) * 10
        DYN = torch.export.Dim.DYNAMIC

        ep = export(
            model,
            (x, y),
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN, 1: DYN}),
            strict=True,
        )
        self.assertTrue(torch.allclose(ep.module()(x, y), model(x, y)))
        x2 = torch.arange(4).reshape((2, 2))
        y2 = torch.arange(9).reshape((3, 3))
        self.assertTrue(torch.allclose(ep.module()(x2, y2), model(x2, y2)))

    @testing.expectedFailureLegacyExportNonStrict  # Some small change due to unbacked values getting regenerated
    @testing.expectedFailureLegacyExportStrict  # Some small change due to unbacked values getting regenerated
    def test_export_max_nonstrict(self):
        class FooMax(torch.nn.Module):
            def forward(self, x):
                return torch.ones(max(x.item(), 1024))

        ep_non_strict_foo_max_symint = export(
            FooMax(), (torch.tensor(4),), strict=False
        ).graph
        FileCheck().check_count("torch.sym_max", count=1, exactly=True).run(
            str(ep_non_strict_foo_max_symint)
        )

        class FooMaxTensors(torch.nn.Module):
            def forward(self, x):
                return torch.ones(max(x, x)) + torch.ones(min(x, x))

        ep_non_strict_foo_max_symint = export(
            FooMaxTensors(), (torch.tensor(4),), strict=False
        ).graph
        FileCheck().check_count(
            "torch.ops.aten.maximum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))
        FileCheck().check_count(
            "torch.ops.aten.minimum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))

        class FooMaxTensorsIter(torch.nn.Module):
            def forward(self, x):
                return max([x, x]) + min([x, x]) + max(x, 5) + min(x, 3)

        ep_non_strict_foo_max_symint = export(
            FooMaxTensorsIter(), (torch.tensor(4),), strict=False
        ).graph
        FileCheck().check_count(
            "torch.ops.aten.maximum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))
        FileCheck().check_count(
            "torch.ops.aten.minimum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))
        FileCheck().check_count(
            "torch.ops.aten.clamp.default", count=2, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))

        class FooMaxTensorsSymInt(torch.nn.Module):
            def forward(self, x, y):
                return max([x.shape[0], y.shape[0], x.shape[0]]) + min(
                    [x.shape[0], y.shape[0], x.shape[0]]
                )

        dynamic_shapes = {
            "x": {0: torch.export.Dim.AUTO},
            "y": {0: torch.export.Dim.AUTO},
        }

        ep_non_strict_foo_max_symint = export(
            FooMaxTensorsSymInt(),
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        ).graph
        FileCheck().check_count("torch.sym_max", count=1, exactly=True).run(
            str(ep_non_strict_foo_max_symint)
        )
        FileCheck().check_count("torch.sym_min", count=1, exactly=True).run(
            str(ep_non_strict_foo_max_symint)
        )

        class FooMaxTensorsSymShape(torch.nn.Module):
            def forward(self, x):
                return max(x, x.shape[0])

        dynamic_shapes = {
            "x": {0: torch.export.Dim.AUTO},
        }

        with self.assertRaisesRegex(
            RuntimeError, "Dynamo failed to run FX node with fake tensors"
        ):
            _ = export(
                FooMaxTensorsSymShape(),
                (torch.randn(4, 4),),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            ).graph

        with self.assertRaisesRegex(
            RuntimeError,
            "Boolean value of Tensor with more than one value is ambiguous",
        ):
            _t = export(
                FooMaxTensorsSymShape(),
                (torch.randn(4, 4),),
                dynamic_shapes=dynamic_shapes,
                strict=False,
            ).graph

    def test_math_pow(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()
                p = min(b, 10)
                p = math.pow(p, 10)
                return y * p

        ep = export(M(), (torch.tensor(5), torch.randn(5)), strict=False)
        FileCheck().check_count("torch.sym_min", count=1, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("operator.pow", count=1, exactly=True).run(
            str(ep.graph)
        )

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
            r"Runtime assertion failed for expression Eq\(Mod\(s27\*s77, s77 \- 1\), 0\)",
        ):
            em.module()(torch.randn(4, 5))

        dim0_x = None
        dim1_x = 2 * torch.export.Dim("_dim1_x", max=4000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        em = torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)
        x = torch.randn(3, 5)
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected.*shape\\[1\\] = 5 to be of the form 2\\*s33, where s33 is an integer",
        ):
            em.module()(x)

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

    @testing.expectedFailureSerDerNonStrict
    @testing.expectedFailureRetraceability
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

    def test_dim_dynamic_divisibility(self):
        class M(torch.nn.Module):
            def forward(self, x):
                if x.size(0) % 2 == 0:
                    return x.clone() * 2
                else:
                    return x.clone() * 0

        input1 = (torch.randn(4),)
        model = M()
        dynamic_shapes = {
            "x": {0: torch.export.Dim.DYNAMIC},
        }
        export(model, input1, dynamic_shapes=dynamic_shapes)

    def test_export_func_with_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, kw1, kw2):
                return arg1 + arg2, kw1 + kw2

        kw_func = Module()
        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

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

    def test_kwargs_reorder(self):
        class M(torch.nn.Module):
            def forward(self, *, x, y, z):
                return x + y + z

        ep = export(
            M(), (), {"z": torch.ones(3), "y": torch.ones(3), "x": torch.ones(3)}
        )
        ep.module()(**{"z": torch.ones(3), "y": torch.ones(3), "x": torch.ones(3)})
        ep.module()(z=torch.ones(3), y=torch.ones(3), x=torch.ones(3))
        ep.module()(x=torch.ones(3), z=torch.ones(3), y=torch.ones(3))

    def test_set_example_inputs(self):
        class M(torch.nn.Module):
            def forward(self, a, *, x, y, z):
                return a, x + y + z

        inp = (
            (torch.ones(3),),
            {"z": torch.ones(3), "y": torch.ones(3), "x": torch.ones(3)},
        )
        ep = export(M(), inp[0], inp[1])
        ep.module()(*ep.example_inputs[0], **ep.example_inputs[1])

        ep.example_inputs = (
            (torch.ones(3),),
            {"x": torch.ones(3), "z": torch.ones(3), "y": torch.ones(3)},
        )
        ep.module()(*ep.example_inputs[0], **ep.example_inputs[1])

        with self.assertRaisesRegex(ValueError, "Example inputs should be a tuple"):
            ep.example_inputs = (torch.ones(3),)

        with self.assertRaisesRegex(ValueError, "Ran into a kwarg keyword mismatch"):
            ep.example_inputs = ((torch.ones(3),), {})

        with self.assertRaisesRegex(ValueError, "Trying to flatten user inputs"):
            ep.example_inputs = (
                (),
                {"x": torch.ones(3), "z": torch.ones(3), "y": torch.ones(3)},
            )

    def test_export_func_with_var_postional_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args):
                return arg1 + args[0], arg2 + args[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    @testing.expectedFailureCppRuntime
    @testing.expectedFailureLegacyExportNonStrict
    @testing.expectedFailureLegacyExportStrict
    def test_export_module(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        torch.ones(3, 4),
                        torch.ones(3, 4),
                    )
                )

            def forward(self, x):
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        model = Foo()
        example_inputs = (torch.randn(3, 4),)
        ep = export(model, example_inputs, strict=False)
        before = list(ep.state_dict.keys())
        ep.run_decompositions()
        after = list(ep.state_dict.keys())
        self.assertEqual(before, after)

    def test_export_func_with_keyword_only_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2):
                return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

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

        ep = export(M(), inputs)
        orig_res = M()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    def test_multidimensional_slicing(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()
                torch._check(b >= 0)
                torch._check(b < y.shape[0])
                return y[0, b]

        if is_non_strict_test(self._testMethodName):
            m = M()
            inp = (torch.tensor(4), torch.ones(10, 10))
            r = m(*inp)

            epm = export(m, inp).module()
            er = epm(*inp)

            self.assertTrue(torch.allclose(er, r))

    def test_sequential_slicing(self):
        # See https://github.com/pytorch/pytorch/issues/137455

        class TestModule1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # seq_last as local variable works
                seq_last = self.seq[1:]
                return seq_last(x)

        class TestModule2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                )
                # seq_last as initialized submodule works
                self.seq_last = self.seq[1:]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.seq_last(x)

        inp = (torch.randn(4, 4),)
        for mod in [TestModule1(), TestModule2()]:
            epm = export(mod, inp).module()
            self.assertTrue(torch.allclose(epm(*inp), mod(*inp)))

    def test_unflatten_isinstance(self):
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
                return self.n(x + 1, True) + self.n(x + 1, False)

        x = torch.zeros(4)
        types = {"n": N}
        ep = export(
            M(),
            (x,),
            preserve_module_call_signature=tuple(types.keys()),
        )
        ufm = torch.export.unflatten(ep)
        self.assertTrue(torch.allclose(ufm(x), x + 5))
        for fqn, mod in ufm.named_modules(remove_duplicate=False):
            if cls := types.get(fqn):
                ty = f"{cls.__module__}.{cls.__qualname__}"
                self.assertTrue(ty, mod.type_name())

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

    def test_unflatten_placeholder_update_child2parent_swap(self):
        class Child(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten.dropout_(x, 0.5, False)  # Applying dropout inplace
                return x - 2

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, x):
                f1 = self.child(x)
                f2 = x * 4
                return f1 + f2

        m = Foo()
        inp = torch.ones(3, 10, dtype=torch.float32)
        orig_result = m(inp)

        if not is_retracebility_test(self._testMethodName):
            inp = torch.ones(3, 10, dtype=torch.float32)
            ep = export(m, (inp,), preserve_module_call_signature=("child",))
            unf = unflatten(ep)
            unf.print_readable()

            inp = torch.ones(3, 10, dtype=torch.float32)
            ep_result = ep.module()(inp)
            self.assertTrue(torch.allclose(ep_result, orig_result))

            unf.set_submodule("child", m.child)
            inp = torch.ones(3, 10, dtype=torch.float32)
            unf_result = unf(inp)
            self.assertTrue(torch.allclose(unf_result, orig_result))

    def test_unflatten_placeholder_update_grandchild2cousin_swap(self):
        class Grandchild(torch.nn.Module):
            def forward(self, x):
                a = x.to(torch.float32)  # .to is considered a mutation
                return x + 4, a

        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grandchild = Grandchild()

            def forward(self, x):
                y, a = self.grandchild(x)
                return y + a

        class OtherGrandchild(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OtherChild(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.other_grandchild = OtherGrandchild()

            def forward(self, x):
                return x + self.other_grandchild(x)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.other_child = OtherChild()

            def forward(self, x):
                f1 = self.child(x)
                f2 = self.other_child(x)
                return f1 + f2

        inp = torch.ones(2, 3, dtype=torch.float32)
        orig_result = Foo()(inp)
        self.assertTrue(torch.allclose(orig_result, torch.ones(2, 3) * 9))

        if not is_retracebility_test(self._testMethodName):
            inp = torch.ones(2, 3, dtype=torch.float32)
            ep = export(Foo(), (inp,), preserve_module_call_signature=("child",))
            unf = unflatten(ep)

            inp = torch.ones(2, 3, dtype=torch.float32)
            ep_result = ep.module()(inp)
            self.assertTrue(torch.allclose(ep_result, orig_result))

            unf.set_submodule("child", Child())
            inp = torch.ones(2, 3, dtype=torch.float32)
            unf_result = unf(inp)
            self.assertTrue(torch.allclose(unf_result, orig_result))

    def test_unflatten_buffer_update_child2parent_swap(self):
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.tensor(10))

            def forward(self, x):
                self.buf.add_(1)
                return x + 2

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, x):
                y = self.child(x)  # child.buf <- 10 + 1 = 11, x + 2 = 3
                x = y + self.child.buf  # 14
                y = self.child(x)  # child.buf <- 11 + 1 = 12, x + 2 = 16
                x = y + self.child.buf  # 28
                y = self.child(x)  # child.buf <- 12 + 1 = 13, x + 2 = 30
                x = y + self.child.buf  # 43
                return x

        inp = torch.ones(2, 3, dtype=torch.float32)
        orig_result = Foo()(inp)
        self.assertTrue(torch.allclose(orig_result, torch.ones(2, 3) * 43))

        if not is_retracebility_test(self._testMethodName):
            inp = torch.ones(2, 3, dtype=torch.float32)
            ep = export(Foo(), (inp,), preserve_module_call_signature=("child",))
            unf = unflatten(ep)

            inp = torch.ones(2, 3, dtype=torch.float32)
            ep_result = ep.module()(inp)
            self.assertTrue(torch.allclose(ep_result, orig_result))

            unf.set_submodule("child", Child())
            inp = torch.ones(2, 3, dtype=torch.float32)
            unf_result = unf(inp)
            self.assertTrue(torch.allclose(unf_result, orig_result))

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
    @testing.expectedFailureCppSerDes  # we don't save placeholder metadata
    @testing.expectedFailureSerDerNonStrict
    def test_linear_conv(self):
        strict = True

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

        ep = export(Foo(), (torch.randn(20, 16, 50, 100),), strict=strict)
        for node in ep.graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in ep.graph_signature.inputs_to_buffers
                or node.name in ep.graph_signature.inputs_to_parameters
            ):
                self.assertTrue("source_fn_stack" in node.meta)

    def test_dynamic_shapes_dataclass(self):
        torch.export.register_dataclass(
            Inp2,
            serialized_type_name="test_export_api_with_dynamic_shapes.Inp2",
        )

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs.a, inputs.b)

        foo = Foo()
        inputs = (Inp2(a=torch.randn(10, 2, 3), b=torch.randn(10, 3, 4)),)
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"inputs": [{0: batch}, {0: batch}]},
        )
        self.assertEqual(
            [
                # First dimension varies across strict and non-strict
                # since the source names are different, resulting in
                # different symbol names.
                str(node.meta["val"].shape[1:])
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([2, 3])", "torch.Size([3, 4])"],
        )

    @testing.expectedFailureCppSerDes
    def test_export_method(self):
        from torch._export.utils import sync_state, wrap_method

        strict = True

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.t = torch.nn.Buffer(torch.tensor(10))

            def forward(self, x):
                return self.foo(x) * self.bar(x)

            def foo(self, x):
                self.t.mul_(2)
                return x + self.t

            def bar(self, x):
                return x - self.t

        # exporting...
        em = M()
        ex = torch.randn(4)

        # ...foo
        epm_foo = export(
            wrap_method(em.foo),
            (ex,),
            dynamic_shapes={"x": (Dim.DYNAMIC,)},
            strict=strict,
        ).module()

        # ...bar
        epm_bar = export(
            wrap_method(em.bar),
            (ex,),
            dynamic_shapes=((Dim.DYNAMIC,),),
            strict=strict,
        ).module()

        if is_serdes_test(self._testMethodName):
            sync_state(epm_foo, epm_bar)

        # running...
        m = M()
        rx = torch.randn(5)

        self.assertTrue(torch.allclose(m.t, epm_foo.t))
        self.assertTrue(torch.allclose(m.t, epm_bar.t))

        # ...foo
        self.assertTrue(torch.allclose(epm_foo(rx), m.foo(rx)))
        self.assertTrue(torch.allclose(m.t, epm_foo.t))
        self.assertTrue(torch.allclose(m.t, epm_bar.t))

        # ...bar
        self.assertTrue(torch.allclose(epm_bar(rx), m.bar(rx)))
        self.assertTrue(torch.allclose(m.t, epm_foo.t))
        self.assertTrue(torch.allclose(m.t, epm_bar.t))

    def test_export_api_with_dynamic_shapes(self):
        from torch.export import Dim, dims

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
                "You marked.*but your code specialized it to be a constant.*less strict API such as maybe_mark_dynamic or Dim.AUTO.(.*\n)*.*"
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

        for node in efoo.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(node.meta["val"].shape[1], node.meta["val"].shape[2])
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [multiple, mostly distinct]
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        efoo = export(
            Foo(),
            inputs,
            dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
        )
        placeholders = [
            node.meta["val"].shape
            for node in efoo.graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(
            placeholders[0][2],
            placeholders[1][1],
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
                # First dimension varies across strict and non-strict
                # since the source names are different, resulting in
                # different symbol names.
                str(node.meta["val"].shape[1:])
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([2, 3])", "torch.Size([3, 4])"],
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
                # First dimension varies across strict and non-strict
                # since the source names are different, resulting in
                # different symbol names.
                str(node.meta["val"].shape[1:])
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([2, 3])", "torch.Size([3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

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
                "You marked.*but your code specialized it to be a constant.*less strict API such as maybe_mark_dynamic or Dim.AUTO(.*\n)*"
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
            (
                {"data": torch.randn(4, 4)},
                torch.randn(4, 4),
                torch.randn(6, 4),
            ),
            {
                "a": torch.randn(8, 4),
                "b": torch.randn(9, 6),
            },
        )
        dynamic_shapes = {
            "x": (
                {"data": (Dim("dx00"), Dim("dx01"))},
                (Dim("dx10"), Dim("dx11")),
                (Dim("dx20"), Dim("dx21")),
            ),
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
        self.assertTrue(all(shape == "torch.Size([s3])" for shape in input_shapes))

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

        torch.export.register_dataclass(
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

        torch.export.register_dataclass(
            Inner, serialized_type_name="test_pytree_register_nested_data_class.Inner"
        )
        torch.export.register_dataclass(
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
                "Dynamo does not support RNN, GRU, or LSTM.",
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

        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # aten.to will just specialize by decomposing to a no-op
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                ],
            )

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 1)

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

        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # aten.to will just specialize by decomposing to a no-op
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                ],
            )

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 1)

    def test_device_to_mutation(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.to("cpu")
                y.add_(1)
                return y, x

        ep = export(Module(), (torch.tensor(1, device="cpu"),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # aten.to decomposes to no-op, add_ decomposes to functional variant
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.add.Tensor,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                    torch.ops.aten.add_.Tensor,
                ],
            )

        # test mutation
        x = torch.tensor(2, device="cpu")
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 3)
        self.assertEqual(id(y), id(x))

        # test decomp ep
        ep = ep.run_decompositions({})
        for node in ep.graph.nodes:
            if node.op == "call_function":
                self.assertNotEqual(node.target, torch.ops.aten.to.dtype_layout)

        # test mutation for decomposed program
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 4)
        self.assertEqual(id(y), id(x))

    @requires_gpu
    @testing.expectedFailureCppRuntime
    def test_device_to_gpu(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(Foo(), (torch.randn(64).to(GPU_TYPE),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # aten.to decomposes to _to_copy
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten._to_copy.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                ],
            )

        # Check device assertion
        with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
            ep.module()(torch.randn(64))

        ep = ep.run_decompositions()
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 2)
        self.assertEqual(
            ops,
            [
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten._to_copy.default,
            ],
        )

        # Check device assertion again after decomp
        with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
            ep.module()(torch.randn(64))

    def test_tensor_constant_aten_to(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.t = torch.tensor([1.0])

            def forward(self, x):
                return x + self.t.to(torch.float64)

        inputs = (torch.randn(1, 10),)
        model = Module()
        ep = export(model, inputs).run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        self.assertIn(torch.ops.aten._to_copy.default, ops)

        self.assertEqual(ep.module()(*inputs), model(*inputs))

    def test_float_conversion(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # .float() decomposes to no-op
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype,
                ],
            )

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 1)

        # test aliasing
        x = torch.tensor(1, dtype=torch.float)
        out = ep.module()(x)
        self.assertEqual(id(x), id(out))

    def test_float_conversion_from_int(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        ep = export(Module(), (torch.tensor(1, dtype=torch.int32),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # .float() decomposes to _to_copy()
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten._to_copy.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype,
                ],
            )

        # Raises error because the input dtype is not the same as the input
        # tensor when exporting.
        with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
            ep.module()(torch.tensor(1, dtype=torch.float32))

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(
            ops,
            [
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten._to_copy.default,
            ],
        )

        # Check dtype assertion again after decomp
        with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
            ep.module()(torch.tensor(1, dtype=torch.float32))

        self.assertEqual(ep.module()(torch.tensor(1, dtype=torch.int32)), 1)

    def test_device_to_mutation_float(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.float()
                y.add_(1)
                return y, x

        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_legacy_test(self._testMethodName) or is_training_ir_test(
            self._testMethodName
        ):
            # aten.to decomposes to no-op, add_ decomposes to functional variant
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.add.Tensor,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype,
                    torch.ops.aten.add_.Tensor,
                ],
            )

        # test mutation
        x = torch.tensor(2, dtype=torch.float)
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 3.0)
        self.assertEqual(id(y), id(x))

        # test decomp ep
        ep = ep.run_decompositions({})
        for node in ep.graph.nodes:
            if node.op == "call_function":
                self.assertNotEqual(node.target, torch.ops.aten.to.dtype)

        # test mutation for decomposed program
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 4.0)
        self.assertEqual(id(y), id(x))

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

    def test_decomp_item_in_prim_after_decomposition(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "Fail")
                return x

        decomp_table = {**default_decompositions(), **decomposition_table}

        ep = export_for_training(M(), (torch.randn(2, 2),)).run_decompositions(
            decomp_table
        )

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, c_lifted_tensor_0, x):
    clone = torch.ops.prims.clone.default(c_lifted_tensor_0, memory_format = torch.preserve_format);  c_lifted_tensor_0 = None
    _assert_async = torch.ops.aten._assert_async.msg(clone, 'Fail');  clone = _assert_async = None
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

        gm = torch.export.export_for_training(mod, (inp,)).module()
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
    batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, False, 0.1, 1e-05, True);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    return pytree.tree_unflatten((batch_norm,), self._out_spec)""",
        )

        mod.train()
        gm_train = torch.export.export_for_training(mod, (inp,)).module()
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
    add_ = torch.ops.aten.add_.Tensor(bn_num_batches_tracked, 1);  bn_num_batches_tracked = add_ = None
    batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, True, 0.1, 1e-05, True);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    return pytree.tree_unflatten((batch_norm,), self._out_spec)""",
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

    def test_while_loop_simple(self):
        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i > 0

                def body_fn(i, x, y):
                    return i - 1, x + y, y - x

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

        example_inputs = (
            torch.tensor(1),
            torch.randn(10, 20),
            torch.randn(10, 20),
        )
        ep = export(Simple(), example_inputs)
        self.assertEqual(ep.module()(*example_inputs), Simple()(*example_inputs))

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

        ep = export(M(), (torch.tensor(1),))
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

    def test_size_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, theta, size):
                return torch.nn.functional.affine_grid(theta, size, align_corners=None)

        model = Model()
        theta = torch.ones((1, 2, 3))
        size = torch.Size((1, 3, 24, 24))
        inp = (theta, size)
        eager_result = model(*inp)

        ep = export(model, inp)

        epm = ep.module()
        ep_result = epm(*inp)
        self.assertTrue(torch.allclose(ep_result, eager_result))

        args, _kwargs = ep.example_inputs
        self.assertTrue(torch.allclose(arg, i) for arg, i in zip(args, inp))

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
                return torch.randn((a, 4))

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
                randn = torch.randn((a, 4))

                return torch.cat((randn.transpose(0, 1), torch.zeros(6, a)), 0)

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

    def test_module_input(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, m):
                return m(x, y) + x + y

        i = InputModule()
        f = Foo()
        ep = export(f, (torch.randn(3), torch.randn(3), i), strict=False)

        m = InputModule()
        inputs = (torch.randn(3), torch.randn(3), m)
        self.assertEqual(f(*inputs), ep.module()(*inputs))

    def test_module_input_subclasses_parameterization_nested(self):
        class Module(torch.nn.Module):
            def forward(self, x, m):
                return m(x) * 2

        mod = InputModuleWithNestedSubclass()
        f = Module()
        ref_x = torch.randn(2, 2)
        ref_out = f(ref_x, mod)

        ep = torch.export.export_for_training(f, (torch.randn(2, 2), mod), strict=False)
        self.assertEqual(ref_out, ep.module()(ref_x, mod))

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

    def test_sym_or_sym_and(self):
        from torch.fx.experimental.symbolic_shapes import sym_and, sym_or

        class Foo(torch.nn.Module):
            def forward(self, xs):
                u0, u1, u2 = xs.tolist()
                torch._check(sym_or(u0 == 2, u0 == 4, u0 == 6))
                torch._check(sym_and(u1 >= 4, u1 <= 8, u2 == 5))
                return u0 + u1 + u2

        ep = export(Foo(), (torch.tensor([2, 6, 5]),), strict=False)
        ep.module()(torch.tensor([2, 6, 5]))
        ep.module()(torch.tensor([4, 7, 5]))
        ep.module()(torch.tensor([6, 5, 5]))
        with self.assertRaisesRegex(
            RuntimeError, r".* expression Eq\(u0, 2\) \| Eq\(u0, 4\) \| Eq\(u0, 6\) .*"
        ):
            ep.module()(torch.tensor([3, 6, 5]))
        with self.assertRaisesRegex(
            RuntimeError, r".* expression Eq\(u2, 5\) & \(4 <= u1\) & \(u1 <= 8\) .*"
        ):
            ep.module()(torch.tensor([6, 6, 6]))

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
        # serdes deserailizes tuple as list
        if need_serdes_test(self._testMethodName):
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

        else:
            if is_inline_and_install_strict_test(self._testMethodName):
                self.assertExpectedInline(
                    ep.graph_module.code.strip(),
                    """\
def forward(self, b____modules__a____buffers__buffer, x):
    sym_size_int_1 = torch.ops.aten.sym_size.int(x, 0)
    gt = sym_size_int_1 > 4;  sym_size_int_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, b____modules__a____buffers__buffer));  gt = true_graph_0 = false_graph_0 = x = b____modules__a____buffers__buffer = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
                )
            else:
                self.assertExpectedInline(
                    ep.graph_module.code.strip(),
                    """\
def forward(self, b_a_buffer, x):
    sym_size_int_1 = torch.ops.aten.sym_size.int(x, 0)
    gt = sym_size_int_1 > 4;  sym_size_int_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, b_a_buffer));  gt = true_graph_0 = false_graph_0 = x = b_a_buffer = None
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

        ep = export(M(), (torch.ones(6, 4),)).run_decompositions({})
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

    @requires_cuda
    @testing.expectedFailureCppRuntime
    def test_export_associative_scan_symbol_dim(self):
        device = torch.device("cuda")
        combine_mode = "pointwise"

        dim1 = torch.export.Dim("dim0", min=5, max=15)
        xs = torch.ones(3, 10, 2, device=device)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def combine_fn(self, x, y):
                return x + y

            def forward(self, x):
                return associative_scan(
                    self.combine_fn, x, 2, combine_mode=combine_mode
                )

        ep = export(Foo(), (xs,), dynamic_shapes={"x": {1: dim1}})
        module_out = Foo()(xs)
        self.assertTrue(torch.allclose(ep.module()(xs), module_out))

    @requires_cuda
    @testing.expectedFailureCppRuntime
    def test_export_associative_scan_symbol_scandim(self):
        device = torch.device("cuda")
        combine_mode = "pointwise"

        dim1 = torch.export.Dim("dim0", min=5, max=15)
        xs = torch.ones(3, 10, 2, device=device)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def combine_fn(self, x, y):
                return x + y

            def forward(self, x):
                return associative_scan(
                    self.combine_fn, x, 1, combine_mode=combine_mode
                )

        ep = export(Foo(), (xs,), dynamic_shapes={"x": {1: dim1}})
        module_out = Foo()(xs)
        self.assertTrue(torch.allclose(ep.module()(xs), module_out))

    @requires_cuda
    @testing.expectedFailureCppRuntime
    def test_export_associative_scan_lifted_buffers(self):
        device = torch.device("cuda")
        combine_mode = "pointwise"

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "buf", torch.ones(3, 2, device=device), persistent=False
                )

            def combine_fn(self, x, y):
                return x + y * self.buf

            def forward(self, x):
                return associative_scan(
                    self.combine_fn, x, 1, combine_mode=combine_mode
                )

        inp = torch.ones(3, 10, 2, device=device)
        ep = export(M(), (inp,))
        epm = ep.module()

        self.assertTrue(torch.allclose(epm(inp), M()(inp)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]),
                1,
            )

    # scan is not supported in sigmoid yet
    @testing.expectedFailureCppRuntime
    def test_export_scan_pytree_output(self):
        def add(carry, accum):
            return carry + carry, (accum[0]["moo"] + 1, accum[0]["moo2"] + 1)

        class M(torch.nn.Module):
            def forward(self, init, accum):
                return scan(add, init, accum)

        inp = torch.randn(3)
        init, xs = torch.ones(3), ({"moo": torch.ones(3), "moo2": torch.ones(3)},)
        ep = export(M(), (init, xs))
        self.assertEqual(ep.module()(init, xs), M()(init, xs))

    # map_fn references module outside the module hierarchy
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

    def test_no_check_is_size_error(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                return torch.randn(24).view(a, 4)

        f = Module()
        ep = export(f, (torch.tensor(6),))
        ep.module()(torch.tensor(6))
        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for .* u.* 6"
        ):
            ep.module()(torch.tensor(5))

    def test_is_non_negative_check_function(self):
        import sympy as sp

        from torch.fx.experimental.symbolic_shapes import _is_non_negative_check

        x = sp.Symbol("x")
        variable_name = sp.Symbol("variable_name")
        tensor_shape = sp.Symbol("tensor.shape[0]")

        self.assertEqual(_is_non_negative_check(variable_name >= 0), "variable_name")
        self.assertEqual(_is_non_negative_check(tensor_shape >= 0), "tensor.shape[0]")

        # Test cases where the condition is not checking for x >= 0
        self.assertIsNone(_is_non_negative_check(x > 0))
        self.assertIsNone(_is_non_negative_check(x == 0))
        self.assertIsNotNone(_is_non_negative_check(0 <= x))
        self.assertIsNone(_is_non_negative_check(x >= 1))

    def test_suggest_torch_checks_with_non_negative_check(self):
        from unittest.mock import patch

        import sympy

        from torch.export.dynamic_shapes import defaultdict
        from torch.fx.experimental.symbolic_shapes import _suggest_torch_checks

        u = sympy.Symbol("u")
        cond = u >= 0
        mock_exception = MagicMock(
            spec=torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        )
        mock_exception.args = ["Test error message"]
        mock_exception.cond = cond

        mock_printer = MagicMock()
        mock_printer.doprint.side_effect = lambda expr: (
            str(cond) if expr == cond else "u < 0"  # Simulating the condition
        )
        with patch(
            "torch.fx.experimental.symbolic_shapes._PythonMsgPrinter",
            return_value=mock_printer,
        ):
            src_map = defaultdict(list)
            src_map["u"] = ["u"]
            _suggest_torch_checks(mock_exception, src_map)
            error_msg = mock_exception.args[0]
            self.assertIn("torch._check_is_size(u)", error_msg)
            self.assertIn("torch._check(u < 0)", error_msg)

    def test_suggest_torch_checks_with_regular_check(self):
        import sympy

        from torch.export.dynamic_shapes import defaultdict
        from torch.fx.experimental.symbolic_shapes import _suggest_torch_checks

        mock_exception = MagicMock(
            spec=torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        )
        mock_exception.args = ["Test error message"]

        mock_cond = MagicMock()
        mock_cond.free_symbols = {sympy.Symbol("u")}
        mock_exception.cond = mock_cond

        mock_printer = MagicMock()
        mock_printer.doprint.side_effect = lambda expr: (
            "u > 5" if expr == mock_cond else "u <= 5"
        )

        with patch(
            "torch.fx.experimental.symbolic_shapes._PythonMsgPrinter",
            return_value=mock_printer,
        ):
            src_map = defaultdict(list)
            src_map["u"] = ["u"]

            _suggest_torch_checks(mock_exception, src_map)

            error_msg = mock_exception.args[0]
            self.assertIn("torch._check(u > 5)", error_msg)
            self.assertIn("torch._check(u <= 5)", error_msg)
            self.assertNotIn("torch._check_is_size", error_msg)

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

        class Boo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(True)

            def forward(self, x):
                list_tensor = [torch.tensor(False), torch.tensor(True)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Boo(), (torch.tensor(False),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 3)

        inp = (torch.tensor(True),)
        self.assertTrue(torch.allclose(ep.module()(*inp), Boo()(*inp)))

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
        torch.export.register_dataclass(
            Inp3,
            serialized_type_name="test_preserve_shape_dynamism_for_unused_inputs.Inp3",
        )

        class Module(torch.nn.Module):
            def forward(self, x: Inp3):
                return x.f + 1

        mod = Module()
        example_inputs = (Inp3(f=torch.ones(10, 4), p=torch.zeros(10, 4)),)
        ep_static = export(mod, example_inputs)
        for node in ep_static.graph.nodes:
            if node.op == "placeholder":
                for s in node.meta["val"].shape:
                    self.assertIsInstance(s, int)

        dim0_x_f, dim0_x_p = torch.export.dims("dim0_x_f", "dim0_x_p")
        dynamic_shapes = {"x": [{0: dim0_x_f}, {0: dim0_x_p}]}
        ep_dynamic = export(mod, example_inputs, dynamic_shapes=dynamic_shapes)
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

    def test_multinomial_dynamic(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.multinomial(x, y.shape[0])

        model = Model()
        DYNAMIC = torch.export.Dim.DYNAMIC

        def exported_module(inputs):
            dynamic_shapes = tuple(tuple(DYNAMIC for _ in inp.shape) for inp in inputs)
            ep = export(model, inputs, dynamic_shapes=dynamic_shapes)
            return ep.module()

        def check(inputs, epm):
            eager_result = model(*inputs)
            ep_result = epm(*inputs)
            self.assertEqual(ep_result.shape, eager_result.shape)

        inputs = (
            torch.tensor([0, 10, 3, 0], dtype=torch.float32),
            torch.ones(2, dtype=torch.int64),
        )
        epm = exported_module(inputs)
        # output shape is (2,), where n_sample 2 <= dist_size 4
        check(inputs, epm)

        inputs = (
            torch.tensor([0, 10, 3, 7, 6, 0], dtype=torch.float32),
            torch.ones(3, dtype=torch.int64),
        )
        # output shape is (3,), with n_sample 3 <= dist_size 6
        check(inputs, epm)

        inputs = (
            torch.tensor([0, 10, 3, 0], dtype=torch.float32),
            torch.ones(5, dtype=torch.int64),
        )
        with self.assertRaisesRegex(RuntimeError, "cannot sample"):
            # n_sample 5 > dist_size 4
            epm(*inputs)

        inputs = (
            torch.tensor([[4, 5], [6, 7], [8, 9]], dtype=torch.float32),
            torch.ones(2, dtype=torch.int64),
        )
        epm = exported_module(inputs)
        # output shape is (3, 2), with n_row 3 and n_sample 2 <= dist_size 2
        check(inputs, epm)

        inputs = (
            torch.tensor([[4, 5], [6, 7], [8, 9]], dtype=torch.float32),
            torch.ones(3, dtype=torch.int64),
        )
        epm = exported_module(inputs)
        with self.assertRaisesRegex(RuntimeError, "cannot sample"):
            # n_sample 3 > dist_size 2
            epm(*inputs)

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

        if is_inline_and_install_strict_test(self._testMethodName):
            self.assertExpectedInline(
                str(ep_decompose_linear.graph_module.code).strip(),
                """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_bias, c_linear_weight, x, y):
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

        else:
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

    # TODO(pianpwk) blocker: https://github.com/pytorch/pytorch/issues/151809
    @testing.expectedFailureSerDer
    @testing.expectedFailureSerDerNonStrict
    @testing.expectedFailureCppSerDes
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
        ep = export_for_training(M2(), inps).run_decompositions({})
        self.assertTrue(torch.allclose(ep.module()(*inps), M2()(*inps)))

        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 1)
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %clone), kwargs = {})
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
        ep = export_for_training(M2(), inps, strict=False).run_decompositions({})
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
    %detach_1 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach,), kwargs = {})
    %detach_2 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_1,), kwargs = {})
    %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %detach_3 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%clone,), kwargs = {})
    %detach_4 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_3,), kwargs = {})
    %detach_5 : [num_users=1] = call_function[target=torch.ops.aten.detach.default](args = (%detach_4,), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%detach_2, %detach_5), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %x), kwargs = {})
    return (mul_1,)""",
        )

        unflattened = unflatten(ep)
        self.assertTrue(torch.allclose(unflattened(*inps), M2()(*inps)))

    def test_module_dict_key(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = torch.nn.Linear(10, 10)

            def forward(self, x, d):
                d = {m: d[name] for name, m in self.named_children()}
                return x + d[self.mod]

        m = Module()
        sample_inputs = (torch.randn(10), {"mod": torch.randn(10)})
        ep = export(m, sample_inputs)
        self.assertEqual(ep.module()(*sample_inputs), m(*sample_inputs))

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

    def test_operator_aten_tensor_mode_variant(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.div.Tensor_mode(x, 2, rounding_mode="floor")

        m = Module()
        args = (torch.randn(4, 3),)
        ep = export(m, args)
        self.assertEqual(ep.module()(*args), m(*args))

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
        ep_strict = export(mod, inp)
        ep_non_strict = export(mod, inp, strict=False)

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

    def test_unflatten_random_dag_5(self):
        # dag: {0: [1, 2, 3], 1: [2, 4], 2: [4], 3: [], 4: []}

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n4 = N4()

            def forward(self, x):
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n3 = N3()

            def forward(self, x):
                x = self.n3.n4(x + 1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n2 = N2()

            def forward(self, x):
                x = self.n2(x + 1)
                x = self.n2.n3.n4(x + 1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = N1()

            def forward(self, x):
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                return x + 1

        n0 = N0()
        inp = (torch.ones(1),)
        eager = n0(*inp)
        ep = export(n0, inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        self.assertTrue(torch.allclose(epm(*inp), eager))
        self.assertTrue(torch.allclose(ufm(*inp), eager))

    def test_unflatten_random_dag_6(self):
        # dag: {0: [1, 2, 4, 5], 1: [3, 5], 2: [4, 5], 3: [], 4: [5], 5: []}

        class N5(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n5 = N5()

            def forward(self, x):
                x = self.n5(x + 1)
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n4 = N4()

            def forward(self, x):
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n3 = N3()

            def forward(self, x):
                x = self.n3.n4(x + 1)
                x = self.n3.n4.n5(x + 1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n2 = N2()

            def forward(self, x):
                x = self.n2.n3(x + 1)
                x = self.n2.n3.n4.n5(x + 1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = N1()

            def forward(self, x):
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3.n4(x + 1)
                x = self.n1.n2.n3.n4.n5(x + 1)
                return x + 1

        n0 = N0()
        inp = (torch.ones(1),)
        eager = n0(*inp)
        ep = export(n0, inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        self.assertTrue(torch.allclose(epm(*inp), eager))
        self.assertTrue(torch.allclose(ufm(*inp), eager))

    def test_unflatten_random_dag_buf_8(self):
        class N7(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N6(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n7 = N7()

            def forward(self, x):
                x = self.n7(x + 1)
                return x + 1

        class N5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n6 = N6()

            def forward(self, x):
                x = x + self.n6.n7.buf
                x = self.n6(x + 1)
                return x + 1

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n5 = N5()

            def forward(self, x):
                x = x + self.n5.buf
                x = self.n5(x + 1)
                x = self.n5.n6(x + 1)
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n4 = N4()

            def forward(self, x):
                x = x + self.n4.buf
                x = x + self.n4.n5.n6.n7.buf
                x = self.n4(x + 1)
                x = self.n4.n5.n6(x + 1)
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.n4.n5.n6.n7.buf
                x = self.n3(x + 1)
                x = self.n3.n4(x + 1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.n3.n4.n5.buf
                x = x + self.n2.n3.n4.n5.n6.n7.buf
                x = self.n2(x + 1)
                x = self.n2.n3.n4(x + 1)
                x = self.n2.n3.n4.n5(x + 1)
                x = self.n2.n3.n4.n5.n6.n7(x + 1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.buf
                x = x + self.n1.n2.n3.n4.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3.n4(x + 1)
                x = self.n1.n2.n3.n4.n5.n6.n7(x + 1)
                return x + 1

        n0 = N0()
        inp = (torch.ones(1),)
        eager = n0(*inp)
        ep = export(n0, inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_4(self):
        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                x = self.n3(x + 1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.n3.buf
                x = self.n2(x + 1)
                x = self.n2.n3(x + 1)
                self.n2.n3.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.buf
                x = x + self.n1.n2.n3.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                self.n1.buf.add_(1)
                self.n1.n2.buf.add_(1)
                return x + 1

        n0 = N0()
        inp = (torch.ones(1),)
        eager = n0(*inp)
        ep = export(N0(), inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_6(self):
        class N5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n5 = N5()

            def forward(self, x):
                x = x + self.n5.buf
                self.n5.buf.add_(1)
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n4 = N4()

            def forward(self, x):
                x = x + self.n4.buf
                x = self.n4(x + 1)
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                x = x + self.n3.n4.n5.buf
                x = self.n3(x + 1)
                x = self.n3.n4(x + 1)
                x = self.n3.n4.n5(x + 1)
                self.n3.n4.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.n3.n4.buf
                x = self.n2.n3(x + 1)
                x = self.n2.n3.n4(x + 1)
                x = self.n2.n3.n4.n5(x + 1)
                self.n2.buf.add_(1)
                self.n2.n3.buf.add_(1)
                self.n2.n3.n4.n5.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.buf
                x = x + self.n1.n2.n3.buf
                x = x + self.n1.n2.n3.n4.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                x = self.n1.n2.n3.n4(x + 1)
                x = self.n1.n2.n3.n4.n5(x + 1)
                self.n1.n2.buf.add_(1)
                self.n1.n2.n3.n4.buf.add_(1)
                return x + 1

        n0 = N0()
        inp = (torch.ones(1),)
        eager = n0(*inp)
        ep = export(N0(), inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_9(self):
        class N8(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N7(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n8 = N8()

            def forward(self, x):
                x = self.n8(x + 1)
                self.n8.buf.add_(1)
                return x + 1

        class N6(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n7 = N7()

            def forward(self, x):
                x = x + self.n7.buf
                x = x + self.n7.n8.buf
                x = self.n7.n8(x + 1)
                self.n7.buf.add_(1)
                self.n7.n8.buf.add_(1)
                return x + 1

        class N5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n6 = N6()

            def forward(self, x):
                x = x + self.n6.n7.buf
                x = self.n6.n7(x + 1)
                self.n6.buf.add_(1)
                self.n6.n7.n8.buf.add_(1)
                return x + 1

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n5 = N5()

            def forward(self, x):
                x = x + self.n5.buf
                x = x + self.n5.n6.buf
                x = self.n5(x + 1)
                x = self.n5.n6.n7(x + 1)
                x = self.n5.n6.n7.n8(x + 1)
                self.n5.n6.n7.buf.add_(1)
                self.n5.n6.n7.n8.buf.add_(1)
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n4 = N4()

            def forward(self, x):
                x = x + self.n4.buf
                x = x + self.n4.n5.n6.n7.buf
                x = x + self.n4.n5.n6.n7.n8.buf
                x = self.n4(x + 1)
                x = self.n4.n5.n6(x + 1)
                self.n4.n5.n6.n7.buf.add_(1)
                self.n4.n5.n6.n7.n8.buf.add_(1)
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.n4.n5.n6.buf
                x = x + self.n3.n4.n5.n6.n7.buf
                x = self.n3(x + 1)
                x = self.n3.n4(x + 1)
                x = self.n3.n4.n5(x + 1)
                x = self.n3.n4.n5.n6.n7.n8(x + 1)
                self.n3.n4.buf.add_(1)
                self.n3.n4.n5.buf.add_(1)
                self.n3.n4.n5.n6.buf.add_(1)
                self.n3.n4.n5.n6.n7.n8.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.n3.buf
                x = x + self.n2.n3.n4.n5.buf
                x = x + self.n2.n3.n4.n5.n6.buf
                x = x + self.n2.n3.n4.n5.n6.n7.n8.buf
                x = self.n2(x + 1)
                x = self.n2.n3.n4(x + 1)
                self.n2.buf.add_(1)
                self.n2.n3.n4.n5.n6.buf.add_(1)
                self.n2.n3.n4.n5.n6.n7.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.buf
                x = x + self.n1.n2.n3.n4.buf
                x = x + self.n1.n2.n3.n4.n5.n6.n7.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                x = self.n1.n2.n3.n4(x + 1)
                x = self.n1.n2.n3.n4.n5.n6.n7(x + 1)
                self.n1.n2.n3.buf.add_(1)
                self.n1.n2.n3.n4.n5.n6.buf.add_(1)
                self.n1.n2.n3.n4.n5.n6.n7.n8.buf.add_(1)
                return x + 1

        n0 = N0()
        inp = (torch.ones(1),)
        eager = n0(*inp)
        ep = export(N0(), inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_preserving_4(self):
        # {0: [1, 2, 3], 1: [2], 2: [], 3: []}
        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n3 = N3()

            def forward(self, x):
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n2 = N2()

            def forward(self, x):
                x = self.n2(x + 1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = N1()

            def forward(self, x):
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
            "n1.n2.n3",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_preserving_4(self):
        # {0: [2, 3], 1: [2], 2: [3], 3: []}
        # {0: [], 1: [3], 2: [3], 3: []}
        # {0: [2, 3], 1: [2], 2: [3], 3: []}
        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                x = self.n3(x + 1)
                self.n3.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.buf
                x = self.n2(x + 1)
                self.n2.n3.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.buf
                x = x + self.n1.n2.n3.buf
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
            "n1.n2.n3",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_preserving_4_1(self):
        # {0: [2], 1: [3], 2: [3], 3: []}
        # {0: [2, 3], 1: [3], 2: [3], 3: []}
        # {0: [1], 1: [3], 2: [], 3: []}
        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                self.n3.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.n3.buf
                x = self.n2.n3(x + 1)
                self.n2.n3.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.buf
                x = self.n1(x + 1)
                self.n1.n2.buf.add_(1)
                self.n1.n2.n3.buf.add_(1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
            "n1.n2.n3",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_preserving_5(self):
        # {0: [1, 2, 3], 1: [3, 4], 2: [3, 4], 3: [4], 4: []}
        # {0: [3], 1: [4], 2: [3, 4], 3: [4], 4: []}
        # {0: [1, 2], 1: [2, 3], 2: [3, 4], 3: [], 4: []}
        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n4 = N4()

            def forward(self, x):
                x = x + self.n4.buf
                self.n4.buf.add_(1)
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                x = x + self.n3.n4.buf
                x = self.n3(x + 1)
                x = self.n3.n4(x + 1)
                self.n3.buf.add_(1)
                self.n3.n4.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.n3.buf
                x = x + self.n2.n3.n4.buf
                x = self.n2(x + 1)
                x = self.n2.n3(x + 1)
                self.n2.n3.n4.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.buf
                x = x + self.n1.n2.buf
                x = x + self.n1.n2.n3.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                self.n1.n2.n3.buf.add_(1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
            "n1.n2.n3",
            "n1.n2.n3.n4",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_preserving_7(self):
        # {0: [3, 4, 5, 6], 1: [2, 3, 4, 5, 6], 2: [3, 4, 5], 3: [5, 6], 4: [6], 5: [6], 6: []}
        # {0: [2, 4, 5, 6], 1: [3, 4, 6], 2: [6], 3: [5], 4: [], 5: [], 6: []}
        # {0: [1, 2, 3, 4, 5, 6], 1: [2, 3, 4], 2: [4, 5, 6], 3: [4, 5, 6], 4: [5, 6], 5: [6], 6: []}
        class N6(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n6 = N6()

            def forward(self, x):
                x = x + self.n6.buf
                x = self.n6(x + 1)
                return x + 1

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n5 = N5()

            def forward(self, x):
                x = x + self.n5.n6.buf
                x = self.n5(x + 1)
                x = self.n5.n6(x + 1)
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n4 = N4()

            def forward(self, x):
                x = x + self.n4.n5.buf
                x = x + self.n4.n5.n6.buf
                x = self.n4(x + 1)
                x = self.n4.n5(x + 1)
                x = self.n4.n5.n6(x + 1)
                self.n4.n5.buf.add_(1)
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                x = x + self.n3.n4.buf
                x = x + self.n3.n4.n5.buf
                x = self.n3.n4(x + 1)
                x = self.n3.n4.n5(x + 1)
                x = self.n3.n4.n5.n6(x + 1)
                self.n3.n4.n5.n6.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.buf
                x = x + self.n2.n3.buf
                x = x + self.n2.n3.n4.buf
                x = x + self.n2.n3.n4.n5.buf
                x = x + self.n2.n3.n4.n5.n6.buf
                x = self.n2(x + 1)
                x = self.n2.n3(x + 1)
                x = self.n2.n3.n4(x + 1)
                self.n2.n3.buf.add_(1)
                self.n2.n3.n4.buf.add_(1)
                self.n2.n3.n4.n5.n6.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.n3.buf
                x = x + self.n1.n2.n3.n4.buf
                x = x + self.n1.n2.n3.n4.n5.buf
                x = x + self.n1.n2.n3.n4.n5.n6.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3(x + 1)
                x = self.n1.n2.n3.n4(x + 1)
                x = self.n1.n2.n3.n4.n5(x + 1)
                x = self.n1.n2.n3.n4.n5.n6(x + 1)
                self.n1.n2.buf.add_(1)
                self.n1.n2.n3.n4.buf.add_(1)
                self.n1.n2.n3.n4.n5.buf.add_(1)
                self.n1.n2.n3.n4.n5.n6.buf.add_(1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
            "n1.n2.n3",
            "n1.n2.n3.n4",
            "n1.n2.n3.n4.n5",
            "n1.n2.n3.n4.n5.n6",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_mutating_buf_preserving_10(self):
        class N9(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                return x + 1

        class N8(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n9 = N9()

            def forward(self, x):
                x = x + self.n9.buf
                self.n9.buf.add_(1)
                return x + 1

        class N7(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n8 = N8()

            def forward(self, x):
                x = self.n8(x + 1)
                x = self.n8.n9(x + 1)
                self.n8.buf.add_(1)
                return x + 1

        class N6(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n7 = N7()

            def forward(self, x):
                x = x + self.n7.n8.buf
                x = self.n7(x + 1)
                x = self.n7.n8.n9(x + 1)
                return x + 1

        class N5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n6 = N6()

            def forward(self, x):
                x = x + self.n6.buf
                x = x + self.n6.n7.buf
                x = x + self.n6.n7.n8.buf
                x = self.n6(x + 1)
                x = self.n6.n7.n8.n9(x + 1)
                self.n6.n7.buf.add_(1)
                self.n6.n7.n8.buf.add_(1)
                return x + 1

        class N4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n5 = N5()

            def forward(self, x):
                x = x + self.n5.n6.buf
                x = x + self.n5.n6.n7.n8.buf
                x = x + self.n5.n6.n7.n8.n9.buf
                x = self.n5(x + 1)
                x = self.n5.n6(x + 1)
                x = self.n5.n6.n7.n8(x + 1)
                x = self.n5.n6.n7.n8.n9(x + 1)
                self.n5.buf.add_(1)
                self.n5.n6.buf.add_(1)
                self.n5.n6.n7.buf.add_(1)
                self.n5.n6.n7.n8.buf.add_(1)
                return x + 1

        class N3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n4 = N4()

            def forward(self, x):
                x = x + self.n4.buf
                x = x + self.n4.n5.n6.buf
                x = x + self.n4.n5.n6.n7.buf
                x = x + self.n4.n5.n6.n7.n8.n9.buf
                x = self.n4(x + 1)
                x = self.n4.n5(x + 1)
                x = self.n4.n5.n6(x + 1)
                x = self.n4.n5.n6.n7.n8(x + 1)
                x = self.n4.n5.n6.n7.n8.n9(x + 1)
                self.n4.n5.n6.buf.add_(1)
                self.n4.n5.n6.n7.n8.buf.add_(1)
                return x + 1

        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n3 = N3()

            def forward(self, x):
                x = x + self.n3.buf
                x = x + self.n3.n4.buf
                x = x + self.n3.n4.n5.n6.n7.n8.buf
                x = self.n3(x + 1)
                x = self.n3.n4(x + 1)
                x = self.n3.n4.n5(x + 1)
                x = self.n3.n4.n5.n6(x + 1)
                x = self.n3.n4.n5.n6.n7.n8(x + 1)
                x = self.n3.n4.n5.n6.n7.n8.n9(x + 1)
                self.n3.buf.add_(1)
                self.n3.n4.buf.add_(1)
                self.n3.n4.n5.buf.add_(1)
                self.n3.n4.n5.n6.buf.add_(1)
                self.n3.n4.n5.n6.n7.buf.add_(1)
                self.n3.n4.n5.n6.n7.n8.n9.buf.add_(1)
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.buf
                x = x + self.n2.n3.buf
                x = x + self.n2.n3.n4.n5.n6.buf
                x = x + self.n2.n3.n4.n5.n6.n7.n8.buf
                x = self.n2(x + 1)
                x = self.n2.n3(x + 1)
                x = self.n2.n3.n4(x + 1)
                x = self.n2.n3.n4.n5(x + 1)
                x = self.n2.n3.n4.n5.n6(x + 1)
                x = self.n2.n3.n4.n5.n6.n7(x + 1)
                self.n2.buf.add_(1)
                self.n2.n3.n4.n5.n6.n7.n8.buf.add_(1)
                self.n2.n3.n4.n5.n6.n7.n8.n9.buf.add_(1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.buf
                x = x + self.n1.n2.n3.buf
                x = x + self.n1.n2.n3.n4.n5.buf
                x = x + self.n1.n2.n3.n4.n5.n6.buf
                x = x + self.n1.n2.n3.n4.n5.n6.n7.buf
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                x = self.n1.n2.n3.n4(x + 1)
                x = self.n1.n2.n3.n4.n5.n6.n7(x + 1)
                x = self.n1.n2.n3.n4.n5.n6.n7.n8(x + 1)
                x = self.n1.n2.n3.n4.n5.n6.n7.n8.n9(x + 1)
                self.n1.n2.n3.buf.add_(1)
                self.n1.n2.n3.n4.n5.n6.n7.n8.buf.add_(1)
                self.n1.n2.n3.n4.n5.n6.n7.n8.n9.buf.add_(1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
            "n1.n2.n3",
            "n1.n2.n3.n4",
            "n1.n2.n3.n4.n5",
            "n1.n2.n3.n4.n5.n6",
            "n1.n2.n3.n4.n5.n6.n7",
            "n1.n2.n3.n4.n5.n6.n7.n8",
            "n1.n2.n3.n4.n5.n6.n7.n8.n9",
        )
        ep = export(
            N0(),
            inp,
            strict=False,  # strict export is slow with large random dags
            preserve_module_call_signature=fqns,
        )
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_unflatten_random_dag_const_preserving_3(self):
        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1)

            def forward(self, x):
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1)
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.const
                x = self.n2(x + 1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1)
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.n2.const
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

    def test_none_buffers(self):
        mod = torch.nn.InstanceNorm1d(1)
        args = (torch.randn(1, 2),)
        ep = torch.export.export(mod, args, strict=False)
        self.assertTrue(torch.allclose(ep.module()(*args), mod(*args)))

    @testing.expectedFailureCppRuntime
    def test_symint_input_basic(self):
        strict = False  # TODO: support strict=True

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        ep = export(M(), (4, 5))
        self.assertEqual(ep.module()(4, 5), 20)

        with self.assertRaisesRegex(RuntimeError, r"to be equal to 4, but got 3"):
            self.assertEqual(ep.module()(3, 6), 18)

        ep = export(
            M(), (4, 5), dynamic_shapes={"x": Dim.DYNAMIC, "y": Dim.AUTO}, strict=strict
        )
        self.assertEqual(ep.module()(4, 5), 20)
        self.assertEqual(ep.module()(3, 6), 18)

        ep = export(
            M(), (5, 5), dynamic_shapes={"x": None, "y": Dim.AUTO}, strict=strict
        )
        self.assertEqual(ep.module()(5, 6), 30)

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x["moo"] * y

        ep = export(
            M(),
            ({"moo": 2}, torch.ones(3, 3)),
            dynamic_shapes={"x": {"moo": Dim.DYNAMIC}, "y": {0: Dim.DYNAMIC}},
            strict=strict,
        )
        inp = ({"moo": 3}, torch.ones(4, 3))
        self.assertTrue(torch.allclose(ep.module()(*inp), M()(*inp)))

    @testing.expectedFailureCppRuntime
    @testing.expectedFailureRetraceability
    @testing.expectedFailureRetraceabilityNonStrict
    def test_symint_input_specialization(self):
        strict = False  # TODO: support strict=True

        class M(torch.nn.Module):
            def forward(self, x, y):
                assert x == 3
                assert y.shape[0] == 4
                return x * y

        inp = (3, torch.randn(4, 4))
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"You marked.*but your code specialized it to be a constant.*less strict API such as maybe_mark_dynamic or Dim.AUTO.",
        ):
            ep = export(
                M(),
                inp,
                dynamic_shapes=(Dim.DYNAMIC, None),
                strict=strict,
            )

        ep = export(
            M(),
            inp,
            dynamic_shapes=(Dim.AUTO, None),
            strict=strict,
        )
        with self.assertRaisesRegex(RuntimeError, "to be equal to 3, but got 4"):
            ep.module()(4, torch.randn(4, 4))

    @testing.expectedFailureCppRuntime
    @testing.expectedFailureRetraceability
    @testing.expectedFailureRetraceabilityNonStrict
    def test_symint_input_ranges(self):
        strict = False  # TODO: support strict=True

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        inp = (3, torch.randn(4, 4))
        ep = export(
            M(),
            inp,
            dynamic_shapes=(Dim.DYNAMIC(min=3, max=10), None),
            strict=strict,
        )

        ep.module()(4, torch.randn(4, 4))
        with self.assertRaisesRegex(RuntimeError, "to be <= 10, but got 16"):
            ep.module()(16, torch.randn(4, 4))
        with self.assertRaisesRegex(RuntimeError, "to be >= 3, but got 2"):
            ep.module()(2, torch.randn(4, 4))

        # While tracing the range was found to be a subset of the original range
        class M(torch.nn.Module):
            def forward(self, x, y):
                assert x > 3
                assert x <= 5
                return x * y

        inp = (4, torch.randn(4, 4))
        ep = export(
            M(),
            inp,
            dynamic_shapes=(Dim.DYNAMIC(min=3, max=10), None),
            strict=strict,
        )
        constraints = list(ep.range_constraints.values())
        self.assertEqual(len(constraints), 1)
        constraint = constraints[0]
        self.assertEqual(constraint.lower, 4)
        self.assertEqual(constraint.upper, 5)

        # While tracing the range was found to be bigger than the original range
        class M(torch.nn.Module):
            def forward(self, x, y):
                assert x > 1
                assert x < 20
                return x * y

        inp = (4, torch.randn(4, 4))
        ep = export(
            M(),
            inp,
            dynamic_shapes=(Dim.DYNAMIC(min=3, max=10), None),
            strict=strict,
        )
        constraints = list(ep.range_constraints.values())
        self.assertEqual(len(constraints), 1)
        constraint = constraints[0]
        self.assertEqual(constraint.lower, 3)
        self.assertEqual(constraint.upper, 10)

        # While tracing the range was found to be outside of the original range
        class M(torch.nn.Module):
            def forward(self, x, y):
                assert x > 10
                assert x < 20
                return x * y

        inp = (14, torch.randn(4, 4))
        with self.assertRaisesRegex(
            ValueError, r"\[3, 10\], conflicting with .* \[11, 19\]"
        ):
            ep = export(
                M(),
                inp,
                dynamic_shapes=(Dim.DYNAMIC(min=3, max=10), None),
                strict=strict,
            )

    @testing.expectedFailureCppRuntime
    def test_symint_input_additional_inputs(self):
        strict = False  # TODO: support strict=True

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        additional_inputs = torch.export.AdditionalInputs()
        additional_inputs.add((5, 5))
        additional_inputs.add((3, 5))
        additional_inputs.add((5, 4))
        ep = torch.export.export(
            M(), (6, 7), dynamic_shapes=additional_inputs, strict=strict
        )
        self.assertEqual(ep.module()(5, 5), 10)
        self.assertEqual(ep.module()(3, 5), 8)
        self.assertEqual(ep.module()(5, 4), 9)

    @testing.expectedFailureCppRuntime
    def test_symint_input_shapes_collection(self):
        strict = False  # TODO: support strict=True

        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        import torch.utils._pytree as pytree
        from torch.export.dynamic_shapes import _IntWrapper

        args = (_IntWrapper(5), _IntWrapper(5))
        shapes_collection = torch.export.ShapesCollection()
        shapes_collection[args[0]] = Dim.DYNAMIC
        shapes_collection[args[1]] = Dim.DYNAMIC
        ep = torch.export.export(
            M(), args, dynamic_shapes=shapes_collection, strict=strict
        )
        self.assertEqual(ep.module()(5, 5), 10)
        self.assertEqual(ep.module()(3, 5), 8)
        self.assertEqual(ep.module()(5, 4), 9)

    def test_unflatten_random_dag_const_preserving_3_1(self):
        class N2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1)

            def forward(self, x):
                return x + 1

        class N1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1)
                self.n2 = N2()

            def forward(self, x):
                x = x + self.n2.const
                x = self.n2(x + 1)
                return x + 1

        class N0(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(1)
                self.n1 = N1()

            def forward(self, x):
                x = x + self.n1.const
                x = self.n1(x + 1)
                x = self.n1.n2(x + 1)
                return x + 1

        inp = (torch.ones(1),)
        eager = N0()(*inp)
        fqns = (
            "n1",
            "n1.n2",
        )
        ep = export(N0(), inp, preserve_module_call_signature=fqns)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        assert torch.allclose(epm(*inp), eager)
        assert torch.allclose(ufm(*inp), eager)

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

        class K(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x0):
                return self.n(x0, True)

        class P(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x):
                x0 = x + 3
                x1 = self.n(x0, True)
                x2 = self.n(x0, False)
                return x1 + x2

        class Q(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k = K()

            def forward(self, x):
                x0 = x + 3
                x1 = self.k.n(x0, True)
                x2 = self.k.n(x0, False)
                return x1 + x2

        class R(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k = K()

            def forward(self, x):
                x0 = x + 3
                x1 = self.k(x0)
                x2 = self.k.n(x0, False)
                return x1 + x2

        class _N(torch.nn.Module):
            def forward(self, x):
                return x + 5

        class _N_1(torch.nn.Module):
            def forward(self, x):
                return x + 6

        for Mod, path_n in [(P, "n"), (Q, "k.n"), (R, "k.n")]:
            m = Mod()
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
                # swapping will not work with retrace
                test(
                    export(Mod(), inp, preserve_module_call_signature=(path_n,)),
                    swap={path_n: N()},
                )

            test(
                export(Mod(), inp),
                swap={path_n: _N(), path_n + "@1": _N_1()},
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
            # swapping will not work with retrace
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

    def test_unflatten_multiple_graphs_dispatch(self):
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
                x = self.n(x, True)
                x = x + 5
                x = self.n(x, False)
                x = x + 6
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

        if is_training_ir_test(self._testMethodName):
            test(
                torch.export.export_for_training(
                    M(),
                    inp,
                    strict=not is_non_strict_test(self._testMethodName),
                    preserve_module_call_signature=("n",),
                )
            )

        test(export(M(), inp, preserve_module_call_signature=("n",)))

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

        def test(ep, swap=None):
            epm = ep.module()
            ufm = torch.export.unflatten(ep)

            exported_result = epm(*inp)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

            if swap:
                for fqn, mod in swap.items():
                    ufm.set_submodule(fqn, mod)
                unflattened_result = ufm(*inp)
                self.assertTrue(torch.allclose(unflattened_result, eager_result))

        if not is_retracebility_test(self._testMethodName):
            # swapping will not work with retrace
            test(
                export(M(), inp, preserve_module_call_signature=("n",)),
                swap={"n": N()},
            )

        test(export(M(), inp))

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

        def test(ep, swap=None):
            epm = ep.module()
            ufm = torch.export.unflatten(ep)

            exported_result = epm(*inp)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened_result = ufm(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

            if swap:
                for fqn, mod in swap.items():
                    ufm.set_submodule(fqn, mod)
                unflattened_result = ufm(*inp)
                self.assertTrue(torch.allclose(unflattened_result, eager_result))

        if not is_retracebility_test(self._testMethodName):
            # swapping will not work with retrace
            test(
                export(M(), inp, preserve_module_call_signature=("n",)),
                swap={"n": N()},
            )
            # running decompositions again should work for all IRs
            ep = export(M(), inp, preserve_module_call_signature=("n",))
            test(ep.run_decompositions({}), swap={"n": N()})

        test(export(M(), inp))

        strict = not is_non_strict_test(self._testMethodName)
        ept = torch.export.export_for_training(
            M(),
            inp,
            strict=strict,
            preserve_module_call_signature=("n",),
        )
        test(ept)

    def test_set_grad_unflatten(self):
        class M1(torch.nn.Module):
            def forward(self, a, b):
                with torch.no_grad():
                    return a + b

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()

            def forward(self, a, b):
                return self.m1(a, b)

        inp = (torch.ones(3, 3), torch.ones(3, 3))
        ep = export(M(), inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        self.assertTrue(torch.allclose(ufm(*inp), epm(*inp)))

    def test_placeholder_update_preserving(self):
        class Child(torch.nn.Module):
            def forward(self, x):
                a = x.add_(3)
                return a - 2

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, x):
                f1 = self.child(x)  # x <- 1 + 3 = 4, x - 2 = 2
                f2 = x * 4  # x * 4 = 16
                return f1 + f2

        inp = torch.ones(2, 3, dtype=torch.float32)
        ep1 = export(Foo(), (inp,))
        inp = torch.ones(2, 3, dtype=torch.float32)
        ep2 = export(Foo(), (inp,), preserve_module_call_signature=("child",))

        inp = torch.ones(2, 3, dtype=torch.float32)
        orig_result = Foo()(inp)

        inp = torch.ones(2, 3, dtype=torch.float32)
        ep1_result = ep1.module()(inp)
        self.assertTrue(torch.allclose(ep1_result, orig_result))
        inp = torch.ones(2, 3, dtype=torch.float32)
        ep2_result = ep2.module()(inp)
        self.assertTrue(torch.allclose(ep2_result, orig_result))

    @testing.expectedFailureLegacyExportNonStrict
    @testing.expectedFailureLegacyExportStrict
    def test_constant_tensor_with_non_functional(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = torch.ones((4, 4, 10))

            def forward(self, x):
                ff = self.params + 2
                ff2 = self.params + 1
                buf = torch.ops.aten.sub_.Tensor(ff, ff2)
                return buf.sum() + x.sum()

        model = TestModel()

        x = torch.zeros((4, 4, 10))

        ep_training = torch.export.export_for_training(model, (x,), strict=False)
        state_dict_before = ep_training.state_dict

        ep = export(model, (x,), strict=False).run_decompositions()
        state_dict_after = ep.state_dict
        self.assertEqual(state_dict_before.keys(), state_dict_after.keys())

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, c_params, x):
    add = torch.ops.aten.add.Tensor(c_params, 2)
    add_1 = torch.ops.aten.add.Tensor(c_params, 1);  c_params = None
    sub = torch.ops.aten.sub.Tensor(add, add_1);  add = add_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(sub, []);  sub = None
    sum_2 = torch.ops.aten.sum.dim_IntList(x, []);  x = None
    add_2 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_2,)""",
        )

    @testing.expectedFailureLegacyExportNonStrict
    @testing.expectedFailureLegacyExportStrict
    def test_constant_tensor_with_non_functional_nested(self):
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = torch.ones((4, 4, 10))

            def forward(self, x):
                return x

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                ff = self.submod.params + 2
                ff2 = self.submod.params + 1
                buf = torch.ops.aten.sub_.Tensor(ff, ff2)
                return buf.sum() + x.sum()

        model = TestModel()

        x = torch.zeros((4, 4, 10))

        ep_training = torch.export.export_for_training(model, (x,), strict=False)
        state_dict_before = ep_training.state_dict

        ep = export(model, (x,), strict=False).run_decompositions()
        state_dict_after = ep.state_dict
        self.assertEqual(state_dict_before.keys(), state_dict_after.keys())

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, c_submod_params, x):
    add = torch.ops.aten.add.Tensor(c_submod_params, 2)
    add_1 = torch.ops.aten.add.Tensor(c_submod_params, 1);  c_submod_params = None
    sub = torch.ops.aten.sub.Tensor(add, add_1);  add = add_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(sub, []);  sub = None
    sum_2 = torch.ops.aten.sum.dim_IntList(x, []);  x = None
    add_2 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_2,)""",
        )

    def test_cond_unflatten(self):
        class M1(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    return x + y

                def false_fn(x, y):
                    return x - y

                return torch.cond(p, true_fn, false_fn, [a, b])

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()

            def forward(self, p, a, b):
                return self.m1(p, a, b)

        inp = (torch.tensor(False), torch.ones(3, 3), torch.ones(3, 3))
        ep = export(M(), inp)
        epm = ep.module()
        ufm = torch.export.unflatten(ep)
        self.assertTrue(torch.allclose(ufm(*inp), epm(*inp)))

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

            ep = export(m, inp, preserve_module_call_signature=("n", "p"))
            exported_result = ep.module()(*inp)
            self.assertTrue(torch.allclose(exported_result, eager_result))

            unflattened = torch.export.unflatten(ep)
            unflattened_result = unflattened(*inp)
            self.assertTrue(torch.allclose(unflattened_result, eager_result))

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
        ).run_decompositions({})
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

    @testing.expectedFailureSerDerNonStrict  # register_constant needs to handle serialization
    @testing.expectedFailureSerDer  # register_constant needs to handle serialization
    def test_register_constant(self):
        @dataclass(frozen=True)
        class MyInput:
            int_1: int
            int_2: int

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, f):
                return x + f.int_1 + f.int_2

        register_constant(MyInput)
        ep = export(Foo(), (torch.randn(2, 2), MyInput(4, 4)), strict=False)

        inp = torch.ones(2, 2)
        self.assertEqual(ep.module()(inp, MyInput(4, 4)), Foo()(inp, MyInput(4, 4)))

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
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (p_bar_linear_bias, p_bar_linear_weight, x));  gt = true_graph_0 = false_graph_0 = p_bar_linear_bias = p_bar_linear_weight = x = None
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

    def test_modules_access_for_deleted_submodule(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.foo = torch.nn.Linear(10, 10)

            def forward(self, x):
                for name, mod in self._modules.items():
                    if mod is None:
                        continue
                    pass
                return self.linear(x)

        mod = Foo()
        mod.foo = None
        mod(torch.randn(10, 10))
        export(mod, (torch.randn(10, 10),), strict=False)

    def test_profiling_code(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                with torch.profiler.record_function("foo"):
                    return x.sin()

        ep = export(Foo(), (torch.randn(5, 5),), strict=True)
        FileCheck().check_count(
            "torch.ops.profiler._record_function_enter_new.default", 0, exactly=True
        ).run(ep.graph_module.code)

    def test_replace_unbacked_with_very_large_upperbound(self):
        strict = True
        # beyond 2^53 where python floats lose precision
        VERY_LARGE_INT = 1000000007999999992

        class Model(torch.nn.Module):
            def forward(self, x, t):
                unbacked = t.item()
                torch._check(unbacked <= VERY_LARGE_INT)

                y = torch.ones(unbacked)
                return x.reshape([-1]) + y

        inp = (
            torch.randn(6, 2),
            torch.tensor([12]),
        )
        spec = {
            "x": (Dim.AUTO, Dim.STATIC),
            "t": (Dim.STATIC,),
        }
        ep = export(Model(), inp, dynamic_shapes=spec, strict=strict)
        self.assertTrue(torch.allclose(Model()(*inp), ep.module()(*inp)))

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
    cond = torch.ops.higher_order.cond(b_pred, true_graph_0, false_graph_0, (b_t, x, y));  b_pred = true_graph_0 = false_graph_0 = b_t = x = y = None
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

    def test_python_asserts_with_sym_int(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                assert y.max().item() > 0
                return y

        model = Model()
        ep = torch.export.export(model, (torch.zeros(4, dtype=torch.int),))

        """
        Graph should look like:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "i32[4]"):
                add: "i32[4]" = torch.ops.aten.add.Tensor(x, 1);  x = None

                max_1: "i32[]" = torch.ops.aten.max.default(add)
                item: "Sym(u0)" = torch.ops.aten.item.default(max_1);  max_1 = None
                ge: "Sym(u0 >= 1)" = item >= 1
                _assert_scalar_default = torch.ops.aten._assert_scalar.default(
                    ge,
                    "Runtime assertion failed for expression u0 >= 1 on node 'ge'"
                );  ge = _assert_scalar_default = None

                gt_1: "Sym(u0 > 0)" = item > 0;  item = None
                _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(
                    gt_1,
                    "Runtime assertion failed for expression 0 < u0 on node 'gt_1'"
                );  gt_1 = _assert_scalar_default_1 = None
                return (add,)
        """
        inputs = (torch.ones(4, dtype=torch.int),)
        self.assertEqual(ep.module()(*inputs), model(*inputs))
        inputs = (-torch.ones(4, dtype=torch.int),)
        with self.assertRaisesRegex(
            RuntimeError, "Runtime assertion failed for expression"
        ):
            ep.module()(*inputs)

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

    def test_module_list_slice(self):
        class ModuleListTruncated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fcs = torch.nn.ModuleList(
                    [torch.nn.Linear(1, 1) for _ in range(2)]
                )

            def forward(self, x):
                for fc in self.fcs[:1]:
                    x = fc(x)
                return x

        x = torch.rand(2, 1)
        mod = ModuleListTruncated()

        epm = export(mod, (x,)).module()
        self.assertTrue(torch.allclose(mod(x), epm(x)))

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

    def test_constant_no_user_inp(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(4, 4)

            def forward(self, x):
                return x.sin()

        a = torch.ones(4, 4)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()
                self.register_buffer("buf", torch.ones(4, 4))

            def forward(self):
                return self.bar(self.bar.a) + a + self.bar.a + self.buf

        export(Foo(), (), strict=False)

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
                a, b = torch.ops.testlib.returns_tensor_symint(x)
                return a, b

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

        ep = export_for_training(M(), inps).run_decompositions({})
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

    def test_placeholder_naming_order(self):
        # See https://github.com/pytorch/pytorch/issues/143732

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(3, 16)
                self.layer2 = torch.nn.Linear(3, 32)

            def forward(self, x1, x2, flag=True):
                x1o = self.layer1(x1)
                x2o = self.layer2(x2)
                return torch.cat([x1o, x2o], dim=1)

        mod = Mod()
        args = (torch.rand(1, 3),)
        kwargs = {"flag": False, "x2": torch.rand(1, 3)}
        ep = export(mod, args, kwargs)

        # check that graph is behaviorally correct
        self.assertTrue(
            torch.allclose(ep.module()(*args, **kwargs), mod(*args, **kwargs))
        )

        # check that graph input names are as expected
        self.assertEqual(ep.graph_signature.user_inputs, ("x1", False, "x2"))

    def test_kwarg_dynamic_shapes_diff_order(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(4, 4)

            def forward(self, baba, *, start, end):
                return baba.sum() + start.sum() + end.sum()

        f = DummyModel()
        kwargs = {
            "end": torch.ones(4, 4, 4),
            "start": torch.ones(4, 4),
        }
        dynamic_shapes = {
            "baba": {0: torch.export.Dim("end_dim")},
            "end": {0: torch.export.Dim("end_dim")},
            "start": {0: torch.export.Dim("end_dim"), 1: torch.export.Dim("end_dim")},
        }
        ep = torch.export.export(
            f, (torch.ones(4, 4),), kwargs, dynamic_shapes=dynamic_shapes
        ).run_decompositions()
        ep.module()(torch.ones(4, 4), **kwargs)

    def test_placeholder_naming_order_variadic(self):
        class Mod(torch.nn.Module):
            def forward(self, a, b, c, **kwargs):
                return a - b + c * kwargs["d"]

        mod = Mod()
        args = (torch.randn(3),)
        kwargs = {"c": torch.randn(3), "b": torch.randn(3), "d": torch.randn(3)}
        ep = export(mod, args, kwargs)
        self.assertTrue(
            torch.allclose(ep.module()(*args, **kwargs), mod(*args, **kwargs))
        )
        self.assertEqual(ep.graph_signature.user_inputs, ("a", "c", "b", "d"))

    def test_isnonzero(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.is_nonzero(x)

        with self.assertRaisesRegex(
            RuntimeError, "Boolean value of Tensor with more than"
        ):
            export(Foo(), (torch.randn(4, 4),), strict=False)

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
        if is_inline_and_install_strict_test(self._testMethodName):
            # when installed, prefix name
            expected_names = [  # user inputs should be prioritized, unprefixed
                ("p____parameters__param", InputKind.PARAMETER),
                ("b____buffers__alpha", InputKind.BUFFER),
                ("b____buffers__beta", InputKind.BUFFER),
                ("c_gamma_1", InputKind.CONSTANT_TENSOR),
                ("p_param", InputKind.USER_INPUT),
                ("b_alpha", InputKind.USER_INPUT),
                ("b_beta", InputKind.USER_INPUT),
                ("c_gamma", InputKind.USER_INPUT),
            ]
        else:
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

    def test_attr_assignment_extra(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                self.bar = x.sum()
                return x + 2

        with self.assertRaisesRegex(
            ValueError,
            "During torch.export, following attrs were created in the model.forward:",
        ):
            _ = export(Foo(), (torch.randn(4, 4),), strict=False)

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
        ep_pre = torch.export.export_for_training(Foo(), inps, strict=False)
        self.assertExpectedInline(
            str(ep_pre.graph_module.submod_1.code).strip(),
            """\
def forward(self, x):
    item = torch.ops.aten.item.default(x);  x = None
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(item);  sym_constrain_range_for_size_default = None
    ge_1 = item >= 3
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 3 on node 'ge_1'");  ge_1 = _assert_scalar_default = None
    le = item <= 5
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 5 on node 'le'");  le = _assert_scalar_default_1 = None
    gt_1 = item > 2
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 2 < u0 on node 'gt_1'");  gt_1 = _assert_scalar_default_2 = None
    lt_1 = item < 6
    _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(lt_1, "Runtime assertion failed for expression u0 < 6 on node 'lt_1'");  lt_1 = _assert_scalar_default_3 = None
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
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u2 >= 3 on node 'ge_1'");  ge_1 = _assert_scalar_default = None
    le_1 = _local_scalar_dense <= 5
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u2 <= 5 on node 'le_1'");  le_1 = _assert_scalar_default_1 = None
    gt = _local_scalar_dense > 2
    _assert_scalar_2 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression 2 < u0 on node 'gt_1'");  gt = _assert_scalar_2 = None
    lt = _local_scalar_dense < 6;  _local_scalar_dense = None
    _assert_scalar_3 = torch.ops.aten._assert_scalar.default(lt, "Runtime assertion failed for expression u0 < 6 on node 'lt_1'");  lt = _assert_scalar_3 = None
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
            r"Runtime assertion failed for expression Eq\(Mod\(s27\*s77, 4\*s77 \- 4\), 0\) on node 'eq.*'",
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
            r"Runtime assertion failed for expression Eq\((.*)\) on node '.*'",
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
            r"Runtime assertion failed for expression Eq\((.*)\) on node '.*'",
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

    def test_unbacked_expand(self):
        class Foo(torch.nn.Module):
            def forward(self, xs):
                u0, u1, u2 = xs.tolist()
                x = torch.empty(u0, u1, 1)
                return x.expand(-1, u1, u2)

        ep = export(Foo(), (torch.tensor([1, 2, 3]),))
        self.assertEqual(
            list(ep.module()(torch.tensor([3, 4, 5])).shape),
            [3, 4, 5],
        )
        self.assertEqual(
            list(ep.module()(torch.tensor([0, 1, 0])).shape),
            [0, 1, 0],
        )

        class Bar(torch.nn.Module):
            def forward(self, xs):
                u0, u1 = xs.tolist()
                x = torch.empty(u0)
                return x.expand(u1)

        ep = export(Bar(), (torch.tensor([2, 2]),))
        self.assertEqual(
            ep.module()(torch.tensor([5, 5])).shape[0],
            5,
        )
        self.assertEqual(
            ep.module()(torch.tensor([1, 1])).shape[0],
            1,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(u0, u1\) .*",
        ):
            ep.module()(torch.tensor([1, 5]))

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
            r"Runtime assertion failed for expression Ne\(s77, 20\)",
        ):
            ep.module()(torch.randn(20, 20, 16))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(Mod\(s77, 20\), 0\)",
        ):
            ep.module()(torch.randn(400, 20, 16))
        ep.module()(torch.randn(42, 20, 16))

    def test_full_on_scalar_tensor(self):
        class Foo(torch.nn.Module):
            def forward(self, val):
                return torch.full((80, 2), val, dtype=torch.float32)

        export(Foo(), args=(torch.tensor(1),))

    def test_custom_pytree(self):
        class Foo:
            def __init__(self, attr1, attr2):
                if attr1 is None:
                    raise ValueError("Shouldn't be None")
                self.attr1 = attr1
                self.attr2 = attr2

        class FooModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo_attr = Foo(torch.ones(4, 4), torch.ones(4, 4))

            def forward(self, foo):
                return foo.attr1.sum() + foo.attr2.sum() + self.foo_attr.attr1.sum()

        def flat(foo):
            return torch.utils._pytree._list_flatten([foo.attr1, foo.attr2])

        def flat_with_keys(foo):
            return torch.utils._pytree._list_flatten_with_keys([foo.attr1, foo.attr2])

        def unflat(val, context):
            l = torch.utils._pytree._list_unflatten(val, context)
            return Foo(l[0], l[1])

        torch.utils._pytree.register_pytree_node(
            Foo,
            flat,
            unflat,
            flatten_with_keys_fn=flat_with_keys,
            serialized_type_name=f"{Foo.__module__}.{Foo.__name__}",
        )

        torch.export.export(
            FooModel(), (Foo(torch.ones(4, 4), torch.ones(4, 4)),), strict=False
        )

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
            r"Runtime assertion failed for expression Ne\(s77, s17\)",
        ):  # fail only at runtime
            ep.module()(torch.randn(4), torch.randn(4))  # fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Ne\(s77, s17\**3\)",
        ):
            ep.module()(torch.randn(64), torch.randn(4))  # fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(s77\**2, 3\*s17\)",
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

    def test_unbacked_kth_value(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                n = y.item()
                k = min(n, 128)
                return x.kthvalue(k, dim=0, keepdim=True).values

        inps = (torch.arange(64), torch.tensor([32]))
        ep = export(Foo(), inps)
        self.assertEqual(ep.module()(*inps).item(), 31)

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

    @testing.expectedFailureLegacyExportNonStrict
    @testing.expectedFailureLegacyExportStrict
    def test_constant_tensor_mutation(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.randn(2, 2)

            def forward(self, x):
                self.foo.add_(5)
                return self.foo + x

        with self.assertRaisesRegex(RuntimeError, "Constant foo is"):
            _ = (
                export(M(), (torch.ones(2, 2),), strict=False)
                .run_decompositions()
                .graph
            )

    def test_constant_return(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.randn(2, 2)

            def forward(self, x):
                return self.foo, self.foo + x

        graph = (
            export(M(), (torch.ones(2, 2),), strict=False).run_decompositions().graph
        )
        self.assertExpectedInline(
            str(graph).strip(),
            """\
graph():
    %c_foo : [num_users=2] = placeholder[target=c_foo]
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%c_foo, %x), kwargs = {})
    return (c_foo, add)""",
        )

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
        ).run_decompositions({})
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
        self.assertEqual(len(repeat_nodes), 0)

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

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureSerDerNonStrict  # construtor is not serialized today
    @testing.expectedFailureSerDer  # constructor is not serialized today
    @testing.expectedFailureRetraceability  # dynamo doesn't work with FlatApply op
    def test_capture_subclass_constructor(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buffer = torch.nn.Buffer(
                    TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
                )

            def forward(self, x):
                two_tensor = TwoTensor(x, TwoTensor(x, x)) + self.buffer
                val = x + two_tensor
                return val.b.a

        mod = Foo()
        ep = export_for_training(mod, (torch.randn(4, 4),), strict=False)
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %b_buffer : [num_users=1] = placeholder[target=b_buffer]
    %x : [num_users=1] = placeholder[target=x]
    %twotensor___init__0 : [num_users=1] = get_attr[target=twotensor___init__0]
    %twotensor_const_func_spec0 : [num_users=1] = get_attr[target=twotensor_const_func_spec0]
    %flat_apply : [num_users=2] = call_function[target=torch.ops.higher_order.flat_apply](args = (%twotensor_const_func_spec0, %twotensor___init__0, %x, %x), kwargs = {})
    %access_subclass_inner_tensor_default_7 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%flat_apply, b), kwargs = {})
    %twotensor___init__1 : [num_users=1] = get_attr[target=twotensor___init__1]
    %twotensor_const_func_spec0_1 : [num_users=1] = get_attr[target=twotensor_const_func_spec0]
    %flat_apply_1 : [num_users=2] = call_function[target=torch.ops.higher_order.flat_apply](args = (%twotensor_const_func_spec0_1, %twotensor___init__1, %access_subclass_inner_tensor_default_7, %flat_apply), kwargs = {})
    %access_subclass_inner_tensor_default_17 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%flat_apply_1, b), kwargs = {})
    %access_subclass_inner_tensor_default_23 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_17, b), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%flat_apply_1, %b_buffer), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%access_subclass_inner_tensor_default_23, %add), kwargs = {})
    %access_subclass_inner_tensor_default_24 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_1, b), kwargs = {})
    %access_subclass_inner_tensor_default_29 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_24, a), kwargs = {})
    return (access_subclass_inner_tensor_default_29,)""",
        )

        inp = torch.randn(4, 4)
        self.assertEqual(ep.module()(inp), mod(inp))

        with torch.inference_mode():
            ep = ep.run_decompositions({})

        # There should be no subclases
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %b_parametrizations_buffer_original0 : [num_users=0] = placeholder[target=b_parametrizations_buffer_original0]
    %b_parametrizations_buffer_original1 : [num_users=1] = placeholder[target=b_parametrizations_buffer_original1]
    %x : [num_users=2] = placeholder[target=x]
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %b_parametrizations_buffer_original1), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %add_1), kwargs = {})
    return (add_4,)""",
        )

        self.assertEqual(ep.module()(inp), mod(inp))

        mod = Foo()
        ep = export(mod, (torch.randn(4, 4),)).run_decompositions({})

        self.assertEqual(ep.module()(inp), mod(inp))
        if is_training_ir_test(self._testMethodName):
            self.assertExpectedInline(
                str(ep.graph).strip(),
                """\
graph():
    %b_parametrizations_buffer_original0 : [num_users=0] = placeholder[target=b_parametrizations_buffer_original0]
    %b_parametrizations_buffer_original1 : [num_users=1] = placeholder[target=b_parametrizations_buffer_original1]
    %x : [num_users=2] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %b_parametrizations_buffer_original1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %add), kwargs = {})
    return (add_1,)""",
            )
        else:
            self.assertExpectedInline(
                str(ep.graph).strip(),
                """\
graph():
    %b_parametrizations_buffer_original0 : [num_users=0] = placeholder[target=b_parametrizations_buffer_original0]
    %b_parametrizations_buffer_original1 : [num_users=1] = placeholder[target=b_parametrizations_buffer_original1]
    %x : [num_users=2] = placeholder[target=x]
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %b_parametrizations_buffer_original1), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %add_1), kwargs = {})
    return (add_4,)""",
            )

    def test_capture_subclass_wrong(self):
        from torch._export.wrappers import (
            mark_subclass_constructor_exportable_experimental,
        )

        with self.assertRaisesRegex(RuntimeError, "on fn which is not supported. If"):

            @torch._disable_dynamo
            @mark_subclass_constructor_exportable_experimental
            def fn(a, b):
                return a + b

        class Foo(torch.nn.Module):
            @torch._disable_dynamo
            @mark_subclass_constructor_exportable_experimental
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        with self.assertRaisesRegex(
            RuntimeError, "TestExport.test_capture_subclass_wrong.<locals>.Foo"
        ):
            export(Foo(), (torch.randn(4, 4),))

    def test_capture_subclass_constructor_torch_ir(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buffer = torch.nn.Buffer(
                    TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
                )

            def forward(self, x):
                two_tensor = TwoTensor(x, TwoTensor(x, x)) + self.buffer
                val = x + two_tensor
                return val.b.a

        mod = Foo()
        gm_torch_ir = _export_to_torch_ir(mod, (torch.randn(4, 4),))
        FileCheck().check_count(
            "torch.testing._internal.two_tensor.TwoTensor", 2, exactly=True
        ).run(gm_torch_ir.code)

    def test_sym_float_operators(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return -(x.max().item() / 2) + x

        m = Module()
        args = (torch.ones(4),)
        ep = export(m, args)
        self.assertEqual(ep.module()(*args), m(*args))

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

    def test_shared_submodule_nn_module_stack(self):
        class Shared(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layernorm = torch.nn.LayerNorm(10)
                self.sub_net = torch.nn.Sequential(
                    layernorm,
                    torch.nn.ReLU(),
                    layernorm,
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return self.sub_net(x)

        eager_module = Shared()
        inps = (torch.rand(10),)
        export_module = export(eager_module, inps, {})

        nn_module_stacks = [
            node.meta.get("nn_module_stack")
            for node in export_module.graph.nodes
            if node.op == "call_function" and "norm" in str(node.target)
        ]
        self.assertEqual(len(nn_module_stacks), 2)
        filtered_nn_module_stack = [
            list(nn_module_stack.values())[-1][0]
            for nn_module_stack in nn_module_stacks
        ]

        if is_inline_and_install_strict_test(self._testMethodName):
            # when inlined and install have same ID so reference same layer
            self.assertEqual(filtered_nn_module_stack[0], "sub_net.0")
            self.assertEqual(filtered_nn_module_stack[1], "sub_net.0")
        else:
            self.assertEqual(filtered_nn_module_stack[0], "sub_net.0")
            self.assertEqual(filtered_nn_module_stack[1], "sub_net.2")

    def test_slice_nn_module_stack(self):
        class N(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()
                self.mod_list_1 = torch.nn.Sequential(*tuple(self.n for _ in range(5)))
                self.mod_list_2 = torch.nn.ModuleList(self.n for _ in range(5))

            def forward(self, x, y):
                for m in self.mod_list_1[2:3]:
                    x = m(x, y)
                for m in self.mod_list_2[4:5]:
                    x = m(x, y)
                return x

        export_module = export(M(), (torch.randn(8), torch.randn(8)))

        nn_module_stacks = [
            node.meta.get("nn_module_stack")
            for node in export_module.graph.nodes
            if node.op == "call_function" and "add" in str(node.target)
        ]
        self.assertEqual(len(nn_module_stacks), 2)
        filtered_nn_module_stack = [
            list(nn_module_stack.values())[-1][0]
            for nn_module_stack in nn_module_stacks
        ]
        if is_inline_and_install_strict_test(self._testMethodName):
            self.assertEqual(filtered_nn_module_stack[0], "mod_list_1.2")
            self.assertEqual(filtered_nn_module_stack[1], "mod_list_1.2")
        else:
            self.assertEqual(
                filtered_nn_module_stack[0], "mod_list_1.slice(2, 3, None).2"
            )
            self.assertEqual(
                filtered_nn_module_stack[1], "mod_list_2.slice(4, 5, None).0"
            )

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
                return (
                    torch.matmul(x, w) + self.b + torch.arange(4, dtype=torch.float16)
                )

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
        # [self.w_pre, self.b]
        lifted_constant_names = list(lifted_constants)
        lifted_constant_values = [lifted_constants[n] for n in lifted_constant_names]
        const_gm, _ = split_const_gm(new_gm, False, lifted_constant_names)
        counter = 0
        for node in const_gm.graph.nodes:
            if node.op == "call_function":
                counter += 1
        self.assertTrue(counter == 4)
        counter = 0
        for n in new_gm.graph.nodes:
            if n.op == "placeholder":
                counter += 1
        # expect 3 existing placeholders and 2 folded constant
        self.assertTrue(counter == 5)
        # return (self.b, folded_const, folded_const)
        const_folded_value = const_gm(*lifted_constant_values)

        test_input = torch.randn(4, 4)
        # new_gm(c_w_pre, b, x, folded_const, folded_const)
        actual = new_gm(
            lifted_constant_values[0],
            const_folded_value[0],
            test_input,
            const_folded_value[1],
            const_folded_value[2],
        )[0]
        expected = mod(test_input)
        self.assertEqual(actual, expected)
        const_gm, _ = split_const_gm(
            ep.graph_module, False, lifted_constant_names, lambda x: True
        )
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
        del decomp_table[torch.ops.testlib.foo_functional.default]

        ep = torch.export.export(M(), (torch.randn(4, 4),)).run_decompositions(
            decomp_table,
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

    def test_run_decompositions_keep_metadata(self):
        """Make sure the metadata is kept after exported program run_decompositions."""

        @torch.library.custom_op("mylib::add", mutates_args=())
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            ...

        @torch.library.register_fake("mylib::add")
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        class TestModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.mylib.add(x, y)

        model = TestModel()
        x_example = torch.randn(2, 3)
        y_example = torch.randn(2, 3)
        exported_program = export(model, (x_example, y_example))

        for node in exported_program.graph.nodes:
            node.meta["custom"] = {"my_field": "dummy"}

        for node in exported_program.graph.nodes:
            self.assertEqual(node.meta["custom"]["my_field"], "dummy")

        decomposed_program = exported_program.run_decompositions()
        for node in decomposed_program.graph.nodes:
            self.assertEqual(node.meta["custom"]["my_field"], "dummy")

    def test_run_decompositions_keep_tensor_constant_metadata(self):
        """Make sure the metadata of tensor constants are kept after run_decompositions."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.ones(3, 3)
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.b + self.linear(x)

        ep = export(M(), (torch.ones(3, 3),))
        for node in ep.graph.nodes:
            node.meta["custom"] = {"my_field": "dummy"}

        for node in ep.graph.nodes:
            self.assertEqual(node.meta["custom"]["my_field"], "dummy")

        decomp_ep = ep.run_decompositions()
        for node in decomp_ep.graph.nodes:
            self.assertEqual(node.meta["custom"]["my_field"], "dummy")

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

        table = torch.export.default_decompositions()
        del table[torch.ops.aten.linear.default]
        ep = ep.run_decompositions(table)

        comp_mod = ep.module()
        inp1 = torch.randn(3, 4)
        inp2 = torch.randn(7, 4)
        self.assertTrue(torch.allclose(comp_mod(inp1), mod(inp1)))
        self.assertTrue(torch.allclose(comp_mod(inp2), mod(inp2)))

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

        torch.export.register_dataclass(
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
            """[[Dim('dx', min=4, max=16)]]""",
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

    # Previously export run_decomp would dispatch
    # sdpa to math backend which doesn't guarantee
    # to return contiguous tensor. As a result, downstream
    # view op would fail. In eager (or normal export), sdpa
    # decomps to flash_attention which has correct handling
    # for non-contiguous output. Since in normal export, we
    # dispatch to flash_attention, we also force run_decomp
    # to follow flash_attention.
    def test_attention(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 768
                self.num_heads = 12
                self.dropout = 0.0
                self.batch_first = True
                self.self_attention = torch.nn.MultiheadAttention(
                    self.embed_dim,
                    self.num_heads,
                    dropout=self.dropout,
                    batch_first=self.batch_first,
                )

            def forward(self, input1: torch.Tensor):
                x, _ = self.self_attention(input1, input1, input1, need_weights=False)
                return x

        inps = (torch.randn(1, 224, 768, device="cpu"),)
        export(Foo(), inps)

    @testing.expectedFailureCppSerDes  # TODO(pianpwk): PowByNatural valuerange deserialization
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
            r"You marked.*but your code specialized it to be a constant.*less strict API such as maybe_mark_dynamic or Dim.AUTO.",
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
        self.assertEqual(num_asserts, 2)
        with self.assertRaises(RuntimeError):
            ep.module()(torch.randn(4, 2))

    @testing.expectedFailureSerDer  # T195866111
    @testing.expectedFailureSerDerNonStrict
    def test_hints_wrapper(self):
        strict = True

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

        ep_for_training = torch.export.export_for_training(M(), (x, y), strict=strict)
        self.assertExpectedInline(
            normalize_gm(
                ep_for_training.graph_module.print_readable(print_output=False)
            ),
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

        ep = export(M(), (x, y), strict=strict).run_decompositions({})
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

    @testing.expectedFailureStrict  # test_hop doesn't have a dynamo implementation
    @testing.expectedFailureRetraceability  # test_hop doesn't have a dynamo implementation
    @testing.expectedFailureTrainingIRToRunDecomp  # test_hop doesn't have a dynamo implementation
    @testing.expectedFailureSerDerNonStrict  # TODO: serde torch.FunctionSchema is not implemented yet
    @testing.expectedFailureSerDer  # TODO: serde torch.FunctionSchema is not implemented yet
    def test_export_function_schema(self):
        import torch.utils._pytree as pytree
        from torch._higher_order_ops.utils import (
            _maybe_run_with_interpreter,
            autograd_not_implemented,
            reenter_make_fx,
            unique_graph_id,
        )
        from torch._ops import HigherOrderOperator
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import (
            ProxyTorchDispatchMode,
            track_tensor_tree,
        )

        pytree.register_constant(torch.FunctionSchema)

        class TestFunctionSchemaHop(HigherOrderOperator):
            def __init__(self):
                super().__init__("test_function_schema")

            def __call__(
                self,
                fn,
                x: torch.Tensor,
                schema: Union[torch.FunctionSchema, pytree.TreeSpec],
            ):
                if isinstance(schema, torch.FunctionSchema):
                    _, schema = pytree.tree_flatten(schema)
                return super().__call__(fn, x, schema)

        def trace_hop(proxy_mode, fn, x, schema):
            sub_gm = reenter_make_fx(fn)(x)
            i, gm_name = unique_graph_id(proxy_mode, prefix="_sub_gm")
            proxy_mode.tracer.root.register_module(gm_name, sub_gm)

            out_proxy = proxy_mode.tracer.create_proxy(
                "call_function",
                test_hop,
                tuple(
                    proxy_mode.tracer.unwrap_proxy(arg) for arg in (sub_gm, x, schema)
                ),
                {},
            )
            example_out = test_hop(sub_gm, x, schema)
            return track_tensor_tree(
                example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
            )

        def dense_hop(fn, x, schema):
            assert isinstance(schema, pytree.TreeSpec)
            schema = pytree.tree_unflatten([], schema)
            assert (
                isinstance(schema, torch.FunctionSchema)
                and schema == torch.ops.aten.sin.default._schema
            )
            return fn(x)

        def fake_hop(mode, fn, x, schema):
            with mode:
                return dense_hop(fn, x, schema)

        def func_hop(ctx, fn, x, schema):
            unwrapped_x = ctx.unwrap_tensors(x)
            functional_fn = ctx.functionalize(_maybe_run_with_interpreter(fn))
            return ctx.wrap_tensors(test_hop(functional_fn, unwrapped_x, schema))

        test_hop = TestFunctionSchemaHop()
        test_hop.py_impl(ProxyTorchDispatchMode)(trace_hop)
        test_hop.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)(dense_hop)
        test_hop.py_impl(FakeTensorMode)(fake_hop)
        test_hop.py_autograd_impl(
            autograd_not_implemented(test_hop, deferred_error=True)
        )
        test_hop.py_functionalize_impl(func_hop)

        class Model(torch.nn.Module):
            def forward(self, x):
                def fn(x):
                    return x.sin()

                return test_hop(fn, x, torch.ops.aten.sin.default._schema)

        mod = Model()
        x = torch.randn(3, 4)
        ep = export(mod, (x,))
        self.assertEqual(x.sin(), ep.module()(x))
        pytree._deregister_pytree_node(torch.FunctionSchema)

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

    @testing.expectedFailureSerDer  # T202237665
    @testing.expectedFailureSerDerNonStrict
    def test_dynamic_sym_round(self):
        class ModuleWithSymRound(torch.nn.Module):
            def forward(self, x):
                out_size = round(x.shape[0] / 2.0)
                return x[:out_size]

        dim_min = 5
        dim_max = 10
        dynamic_shapes = {"x": {0: Dim("n", min=dim_min, max=dim_max)}}

        module = ModuleWithSymRound()
        inp = (torch.randn(8),)
        ep = export(module, inp, dynamic_shapes=dynamic_shapes)

        # Expect builtin round in the export graph
        round_nodes = [
            n for n in ep.graph.nodes if n.op == "call_function" and n.target == round
        ]
        self.assertEqual(len(round_nodes), 1)

        # Check pre/post-export equality
        for i in range(dim_min, dim_max + 1):
            dyn_inp = (torch.randn(i),)
            export_res = ep.module()(*dyn_inp)
            ref_res = module(*dyn_inp)
            self.assertEqual(export_res, ref_res)

    @testing.expectedFailureSerDer
    @testing.expectedFailureSerDerNonStrict
    def test_dynamic_lr_shift(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                rshift = x.shape[0] >> 1
                lshift = x.shape[0] << 1
                return x[:rshift], x[:lshift]

        dynamic_shapes = {"x": {0: Dim("N", min=5, max=10)}}
        inp = (torch.randn(8),)
        ep = export(Module(), inp, dynamic_shapes=dynamic_shapes)
        for op in (operator.lshift, operator.rshift):
            shift_op = [
                n for n in ep.graph.nodes if n.op == "call_function" and n.target == op
            ]
            self.assertEqual(len(shift_op), 1)

    @contextmanager
    def distributed_env(self, world_size):
        try:
            from torch.testing._internal.distributed.fake_pg import FakeStore

            torch.distributed.init_process_group(
                backend="fake",
                world_size=world_size,
                rank=0,
                store=FakeStore(),
            )
            yield

        finally:
            torch.distributed.destroy_process_group()

    @xfailIfDistributedNotSupported
    def test_distributed_all_reduce(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, x):
                y = self.linear(x).abs().clamp(max=1.0) * 2
                torch.distributed.all_reduce(y)
                return y

        with self.distributed_env(world_size=2):
            m = Foo()
            ep = export(m, (torch.randn(4, 4),))
            inp = (torch.randn(4, 4),)
            self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

    @xfailIfDistributedNotSupported
    def test_distributed_all_gather(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                ys = [torch.empty_like(x) for _ in range(2)]
                torch.distributed.all_gather(ys, x)
                return ys

        with self.distributed_env(world_size=2):
            m = Foo()
            ep = export(m, (torch.randn(2),))
            inp = (torch.randn(2),)
            self.assertTrue(
                torch.allclose(a, b) for a, b in zip(ep.module()(*inp), m(*inp))
            )

    @xfailIfDistributedNotSupported
    def test_distributed_all_gather_into_tensor(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.empty(2 * 2)
                torch.distributed.all_gather_into_tensor(y, x)
                return y

        with self.distributed_env(world_size=2):
            m = Foo()
            ep = export(m, (torch.randn(2),))
            inp = (torch.randn(2),)
            self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

    @xfailIfDistributedNotSupported
    @testing.expectedFailureCppRuntime
    def test_distributed_all_to_all_single(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.empty(4)
                torch.distributed.all_to_all_single(y, x)
                return y

        with self.distributed_env(world_size=4):
            m = Foo()
            ep = export(m, (torch.randn(4),))
            nodes = ep.graph.find_nodes(
                op="call_function",
                target=torch.ops._c10d_functional.all_to_all_single.default,
            )
            self.assertEqual(len(nodes), 1)

    @xfailIfDistributedNotSupported
    @testing.expectedFailureCppRuntime
    def test_distributed_reduce_scatter_tensor(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.empty(2)
                torch.distributed.reduce_scatter_tensor(y, x)
                return y

        with self.distributed_env(world_size=2):
            m = Foo()
            ep = export(m, (torch.randn(2 * 2),))
            nodes = ep.graph.find_nodes(
                op="call_function",
                target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            self.assertEqual(len(nodes), 1)

    def test_default_decomposition_core_cia_ops(self):
        """
        Verify that core ATen ops with Composite Implicit Autograd dispatch are not
        decomposed by default.
        """

        # TODO Add avg_pool1d, and adaptive_avg_pool1d when ready.
        # See issue #116684.
        core_cia_ops = {
            "torch.ops.aten.upsample_bilinear2d.vec": (
                torch.ops.aten.upsample_bilinear2d.vec,
                {
                    "align_corners": False,
                    "scale_factors": [2, 2],
                    "output_size": None,
                },
            ),
            "torch.ops.aten.upsample_nearest2d.vec": (
                torch.ops.aten.upsample_nearest2d.vec,
                {
                    "scale_factors": [2, 2],
                    "output_size": None,
                },
            ),
        }

        for op_name, (op, kwargs) in core_cia_ops.items():

            class M(torch.nn.Module):
                def forward(self, x):
                    return op(x, **kwargs)

            ep = export(M(), (torch.randn(2, 3, 4, 5),))
            FileCheck().check_count(op_name, 1, exactly=True).run(ep.graph_module.code)

            decomp_table = default_decompositions()

            ep = ep.run_decompositions(
                decomp_table=decomp_table,
            )
            FileCheck().check_count(op_name, 1, exactly=True).run(ep.graph_module.code)

    def test_wrapper_module(self):
        def f(x):
            return torch.abs(x)

        from torch.export import _wrapper_utils

        model = _wrapper_utils._WrapperModule(f)
        ep = export(
            model,
            (
                torch.randn(
                    8,
                ),
            ),
        )

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, args_0):
    abs_1 = torch.ops.aten.abs.default(args_0);  args_0 = None
    return (abs_1,)""",
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
        # TODO(eqy): this needs to stay in sync with default SDPA priority order
        if (False and SM90OrLater) and not torch.version.hip:
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

    def test_print_graph_signature(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(3))

            def forward(self, x):
                x.add_(1)
                self.buf.add_(2)
                return self.buf + x

        ep = export(M(), (torch.ones(3),))
        self.assertExpectedInline(
            str(ep.graph_signature).strip(),
            """\
# inputs
b_buf: BUFFER target='buf' persistent=True
x: USER_INPUT

# outputs
add: USER_OUTPUT""",
        )

        ep = ep.run_decompositions({})
        self.assertExpectedInline(
            str(ep.graph_signature).strip(),
            """\
# inputs
b_buf: BUFFER target='buf' persistent=True
x: USER_INPUT

# outputs
add_1: BUFFER_MUTATION target='buf'
add: USER_INPUT_MUTATION target='x'
add_2: USER_OUTPUT""",
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

    def test_logging_logger(self):
        strict = True
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

        gm = export(M(), (torch.randn(3, 3),), strict=strict).graph_module
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, x);  x = None
    mul = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul, mul);  mul = None
    return (add, add_1)""",
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
        load_torchbind_test_lib()

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

    def test_export_script_module(self):
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_mod = torch.jit.script(Add())._c

            def forward(self, x, y):
                return self.add_mod.forward(x, y)

        x, y = torch.randn(3, 2), torch.randn(3, 2)
        mod = Mod()
        if is_non_strict_test(self._testMethodName):
            ep = export(mod, (x, y))
            self.assertEqual(ep.module()(x, y), mod(x, y))
            FileCheck().check_count("torch.ops.aten.add.Tensor", 1, exactly=True).run(
                ep.graph_module.code
            )
            return

        # TODO: strict mode doesn't work because dynamo add_mod is treated as a
        # user defined variable. We might need to add a CustomModule variable to support it.
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "UserDefined with non-function"
        ):
            ep = export(mod, (x, y))

    def test_preserve_non_cia_op(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.elu(x)

        ep = export(M(), (torch.randn(2, 3, 4, 5),))
        FileCheck().check_count("torch.ops.aten.elu.default", 1, exactly=True).run(
            ep.graph_module.code
        )

        decomp_table = default_decompositions()

        ep = ep.run_decompositions(
            decomp_table=decomp_table,
        )
        FileCheck().check_count("torch.ops.aten.elu.default", 1, exactly=True).run(
            ep.graph_module.code
        )

    def test_preserve_cia_op(self):
        class StaticResizeTrilinear2dModule(torch.nn.Module):
            def forward(self, x):
                a = torch.nn.functional.interpolate(
                    x,
                    size=(x.shape[2] * 2, x.shape[3] * 3, x.shape[4] * 4),
                    mode="trilinear",
                    align_corners=False,
                    antialias=False,
                )
                return a

        ep = export(StaticResizeTrilinear2dModule(), (torch.randn(2, 3, 4, 5, 6),))
        FileCheck().check_count(
            "torch.ops.aten.upsample_trilinear3d.vec", 1, exactly=True
        ).run(ep.graph_module.code)

        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.upsample_trilinear3d.vec]
        ep = ep.run_decompositions(
            decomp_table=decomp_table,
        )

        FileCheck().check_count(
            "torch.ops.aten.upsample_trilinear3d.vec", 1, exactly=True
        ).run(ep.graph_module.code)


if __name__ == "__main__":
    run_tests()
