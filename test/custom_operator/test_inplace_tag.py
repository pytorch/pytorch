# Owner(s): ["module: custom-operators"]
import torch
from torch import Tensor
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._library.utils import is_inplace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


_test_lib = torch.library.Library("_TestInplaceTag", "DEF")  # noqa: SCOPED_LIBRARY

_test_lib.define(
    "add_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    tags=[torch.Tag.inplace],
)


def _add_inplace_impl(self_: Tensor, other: Tensor) -> Tensor:
    self_.add_(other)
    return self_


_test_lib.impl("add_", _add_inplace_impl, "CompositeExplicitAutograd")
_test_lib.impl("add_", lambda self_, other: self_, "Meta")


@skipIfTorchDynamo("custom operator tests not applicable to dynamo")
class TestInplaceTag(TestCase):
    def setUp(self):
        super().setUp()
        self.lib = torch.library.Library("_TestInplaceTag", "FRAGMENT")  # noqa: SCOPED_LIBRARY

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_basic_inplace(self):
        self.assertTrue(is_inplace(torch.ops._TestInplaceTag.add_.default))

    def test_is_inplace_native(self):
        # Hand-written inplace op
        self.assertTrue(is_inplace(torch.ops.aten.abs_.default))
        self.assertFalse(is_inplace(torch.ops.aten.abs.default))
        # Out op is not inplace
        self.assertFalse(is_inplace(torch.ops.aten.abs.out))
        # Functional op is not inplace
        self.assertFalse(is_inplace(torch.ops.aten.add.Tensor))

    def test_no_positional_args(self):
        with self.assertRaisesRegex(ValueError, "at least one positional argument"):
            self.lib.define(
                "no_args(*, Tensor(a!) out) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_first_arg_not_mutable(self):
        with self.assertRaisesRegex(
            ValueError, "first positional argument to be mutable"
        ):
            self.lib.define(
                "not_mutable(Tensor self, Tensor other) -> Tensor",
                tags=[torch.Tag.inplace],
            )

    def test_first_arg_not_tensor(self):
        with self.assertRaisesRegex(
            ValueError, "first positional argument to be a Tensor"
        ):
            self.lib.define(
                "not_tensor(Tensor(a!)[] self) -> ()",
                tags=[torch.Tag.inplace],
            )

    def test_wrong_return_count(self):
        with self.assertRaisesRegex(ValueError, "must return exactly one value"):
            self.lib.define(
                "no_return(Tensor(a!) self) -> ()",
                tags=[torch.Tag.inplace],
            )

    def test_wrong_return_count_multiple(self):
        with self.assertRaisesRegex(ValueError, "must return exactly one value"):
            self.lib.define(
                "multi_return(Tensor(a!) self) -> (Tensor(a!), Tensor)",
                tags=[torch.Tag.inplace],
            )

    def test_return_not_aliased(self):
        with self.assertRaisesRegex(ValueError, "return the first mutable argument"):
            self.lib.define(
                "bad_alias(Tensor(a!) self) -> Tensor",
                tags=[torch.Tag.inplace],
            )

    def test_return_wrong_alias(self):
        with self.assertRaisesRegex(ValueError, "return the first mutable argument"):
            self.lib.define(
                "wrong_alias(Tensor(a!) self, Tensor(b!) other) -> Tensor(b!)",
                tags=[torch.Tag.inplace],
            )

    def test_additional_mutable_positional_arg(self):
        with self.assertRaisesRegex(
            ValueError, "must only mutate the first positional"
        ):
            self.lib.define(
                "extra_mut(Tensor(a!) self, Tensor(b!) other) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_additional_mutable_kwarg(self):
        with self.assertRaisesRegex(
            ValueError, "must only mutate the first positional"
        ):
            self.lib.define(
                "extra_kwarg(Tensor(a!) self, *, Tensor(b!) out) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_compile_inplace(self, backend):
        def fn(x, y):
            return torch.ops._TestInplaceTag.add_(x, y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        expected = x + y

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled, y)
        self.assertEqual(result, expected)
        self.assertEqual(x_compiled, expected)

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_compile_inplace_duplicated_base(self, backend):
        # add_(x, x): self and other are the same tensor.
        def fn(x):
            return torch.ops._TestInplaceTag.add_(x, x)

        x = torch.randn(3, 4)
        expected = x * 2

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled)
        self.assertEqual(result, expected)
        self.assertEqual(x_compiled, expected)

    def test_compile_inplace_functionalized_graph(self):
        def fn(x, y):
            return torch.ops._TestInplaceTag.add_(x, y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled, y)

        self.assertEqual(result, x + y)
        self.assertEqual(x_compiled, x + y)

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestInplaceTag.add_.default, other = arg1_1, _self_base_index = 0, _all_bases = [arg0_1]);  arg1_1 = None
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
    return (getitem_1,)""",
        )

    def test_compile_inplace_view_functionalized_graph(self):
        def fn(x, y):
            return torch.ops._TestInplaceTag.add_(x[1:3], y)

        x = torch.randn(5, 4)
        y = torch.randn(2, 4)

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x_compiled = x.clone()
        result = compiled_fn(x_compiled, y)

        self.assertEqual(result, x[1:3] + y)
        self.assertEqual(x_compiled[1:3], x[1:3] + y)
        # Unsliced parts are unchanged
        self.assertEqual(x_compiled[0], x[0])
        self.assertEqual(x_compiled[3:], x[3:])

        self.assertExpectedInline(
            backend.fw_graphs[0].code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops._TestInplaceTag.add_.default, other = arg1_1, _self_base_index = 0, _self_slice_dim = 0, _self_slice_start = 1, _self_slice_end = 3, _all_bases = [arg0_1]);  arg1_1 = None
    getitem = auto_functionalized_v2[0]
    getitem_1 = auto_functionalized_v2[1];  auto_functionalized_v2 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
    return (getitem,)""",
        )

    @torch._inductor.config.patch(enable_auto_functionalized_v2=False)
    def test_compile_inplace_uses_v2_hop_when_config_off(self):
        def fn(x, y):
            return torch.ops._TestInplaceTag.add_(x, y)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)

        backend = AotEagerAndRecordGraphs()
        x_compiled = x.clone()
        result = torch.compile(fn, backend=backend, fullgraph=True)(x_compiled, y)

        self.assertEqual(result, x + y)
        self.assertEqual(x_compiled, x + y)
        self.assertIn("auto_functionalized_v2", backend.fw_graphs[0].code)
        self.assertNotIn(
            "torch.ops.higher_order.auto_functionalized(",
            backend.fw_graphs[0].code,
        )


instantiate_parametrized_tests(TestInplaceTag)

if __name__ == "__main__":
    run_tests()
