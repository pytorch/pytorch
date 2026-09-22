# Owner(s): ["module: dynamo"]
from collections import deque
from unittest.mock import patch

import torch
import torch._dynamo.guards
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import config as dc
from torch._dynamo.source import (
    AttrSource,
    CallFunctionNoArgsSource,
    ConstantSource,
    ConstDictKeySource,
    DefaultsSource,
    DictGetItemSource,
    DictSubclassGetItemSource,
    GetItemSource,
    GlobalSource,
    ListGetItemSource,
    LocalSource,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


class RecompileTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_automatic_dynamic_reduce_recompiles(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles(dynamic=None):
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch.compile(foo, backend=cnt, dynamic=dynamic)
            opt(x, y)
            x = torch.randn([3])
            y = torch.randn([3])
            opt(x, y)
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)
            x = torch.randn([5])
            y = torch.randn([5])
            opt(x, y)
            opt(x, y)
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        without = run_foo_6_times_and_count_recompiles(dynamic=False)
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        torch._dynamo.reset()
        with_automatic = run_foo_6_times_and_count_recompiles(dynamic=None)
        self.assertEqual(with_automatic.frame_count, 2)
        self.assertEqual(with_automatic.op_count, 2)
        torch._dynamo.reset()
        with_dynamic = run_foo_6_times_and_count_recompiles(dynamic=True)
        self.assertEqual(with_dynamic.frame_count, 1)
        self.assertEqual(with_dynamic.op_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", True)
    def test_recompiles_true_false_flop(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            if x:
                return y * 2
            else:
                return y * y

        def run_foo_6_times_and_count_recompiles():
            cnt = torch._dynamo.testing.CompileCounter()

            opt = torch.compile(foo, backend=cnt, fullgraph=True)

            x = True
            y = torch.randn([2])
            opt(x, y)
            x = False
            y = torch.randn([2])
            opt(x, y)
            x = True
            y = torch.randn([3])
            opt(x, y)
            x = True
            y = torch.randn([4])
            opt(x, y)
            x = True
            y = torch.randn([5])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_automatic_dynamic_tensor_scalar_change(self):
        # Test the counterfactual, lots of recompiles without this config
        def foo(x, y):
            return x * y

        def run_foo_6_times_and_count_recompiles_swap_types():
            cnt = torch._dynamo.testing.CompileCounter()

            x = torch.randn([2])
            y = torch.randn([2])
            opt = torch.compile(foo, backend=cnt)
            opt(x, y)
            x = torch.randn([3])
            y = 3
            opt(x, y)
            x = torch.randn([4])
            y = torch.randn([4])
            opt(x, y)
            opt(x, y)
            x = torch.randn([5])
            y = 4
            opt(x, y)
            opt(x, y)
            x = torch.randn([6])
            y = torch.randn([6])
            opt(x, y)

            return cnt

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_without_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_with_automatic():
            return run_foo_6_times_and_count_recompiles_swap_types()

        without = run_without_automatic()
        self.assertEqual(without.frame_count, 5)
        self.assertEqual(without.op_count, 5)
        torch._dynamo.reset()
        with_automatic = run_with_automatic()
        self.assertEqual(with_automatic.frame_count, 3)
        self.assertEqual(with_automatic.op_count, 3)

    def test_aliasing_guard_failures(self):
        def foo(a, b, c):
            a.add_(b)
            return c + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch.compile(foo, backend=cnt, fullgraph=True)

        x = torch.randn([3])
        y = torch.randn([3])
        z = torch.randn([3])
        cmp_result = compiled_foo(
            x.detach().clone(), y.detach().clone(), z.detach().clone()
        )
        eager_result = foo(x.detach().clone(), y.detach().clone(), z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        cmp_result = compiled_foo(
            z.detach().clone(), y.detach().clone(), x.detach().clone()
        )
        eager_result = foo(z.detach().clone(), y.detach().clone(), x.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        # No recompile, alias preserved
        self.assertEqual(cnt.frame_count, 1)

        x_clone = x.detach().clone()
        cmp_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        x_clone = x.detach().clone()
        eager_result = compiled_foo(x_clone, y.detach().clone(), x_clone)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def _check_aliasing_guard_failure_with_unavailable_source(self, fn, data):
        failure_reasons = []

        def guard_fail_fn(failure):
            failure_reasons.append(failure.reason)

        cnt = torch._dynamo.testing.CompileCounter()
        # torch.compile does not expose guard_fail_fn.
        compiled_fn = torch._dynamo.optimize(backend=cnt, guard_fail_fn=guard_fail_fn)(
            fn
        )

        t1, t2 = torch.randn(4), torch.randn(4)
        self.assertEqual(compiled_fn(t1, t2, data), fn(t1, t2, data))

        t = torch.randn(4)
        self.assertEqual(compiled_fn(t, t, None), fn(t, t, None))

        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(failure_reasons), 1)
        self.assertIn("Duplicate tensors found", failure_reasons[0])
        self.assertIn(
            "NO_TENSOR_ALIASING guard source(s) no longer evaluate",
            failure_reasons[0],
        )
        return failure_reasons[0]

    def test_aliasing_guard_failure_with_unavailable_list_tensor_source(self):
        def fn(a, b, data):
            x = a + b
            if data is not None:
                x = x + data[0] + data[1]
            return x

        reason = self._check_aliasing_guard_failure_with_unavailable_source(
            fn, [torch.randn(4), torch.randn(4)]
        )

        source_name = "args[2]" if dc.debug_force_nested_calls else "data"
        for source in ("[0]", "[1]"):
            self.assertIn(f"{source_name}{source}", reason)

    def test_aliasing_guard_failure_with_unavailable_dict_tensor_source(self):
        def fn(a, b, data):
            x = a + b
            if data is not None:
                x = x + next(iter(data.values()))
            return x

        reason = self._check_aliasing_guard_failure_with_unavailable_source(
            fn, {object(): torch.randn(4)}
        )
        self.assertIn("dict.keys", reason)
        self.assertIn("TypeError", reason)

    def test_aliasing_guard_recompile_reason_with_unavailable_sources(self):
        class MissingAttr:
            pass

        manager = torch._dynamo.guards.GuardManagerWrapper()
        manager.global_scope = {"G": {}}
        manager.no_tensor_aliasing_sources = [
            LocalSource("duplicate_a"),
            LocalSource("duplicate_b"),
            GlobalSource("missing_global"),
            AttrSource(LocalSource("missing_attr"), "value"),
            GetItemSource(LocalSource("missing_index"), 0),
            DictGetItemSource(LocalSource("missing_key"), "key"),
            DefaultsSource(LocalSource("missing_defaults"), 0),
            GetItemSource(LocalSource("not_subscriptable"), 0),
            ListGetItemSource(LocalSource("not_list"), 0),
            GetItemSource(LocalSource("missing_deque"), 0),
        ]
        duplicate = torch.randn(4)
        scope = {
            "L": {
                "duplicate_a": duplicate,
                "duplicate_b": duplicate,
                "missing_attr": MissingAttr(),
                "missing_index": [],
                "missing_key": {},
                "missing_defaults": lambda: None,
                "not_subscriptable": None,
                "not_list": None,
                "missing_deque": deque(),
            }
        }

        reasons = (
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager, scope
            )
        )

        self.assertEqual(len(reasons), 2)
        self.assertIn("Duplicate tensors found", reasons[0])
        self.assertIn("duplicate_a", reasons[0])
        self.assertIn("duplicate_b", reasons[0])
        self.assertIn(
            "NO_TENSOR_ALIASING guard source(s) no longer evaluate", reasons[1]
        )
        for source, exception_type in (
            ("missing_attr", "AttributeError"),
            ("missing_global", "KeyError"),
            ("missing_index", "IndexError"),
            ("missing_key", "KeyError"),
            ("missing_defaults", "TypeError"),
            ("not_subscriptable", "TypeError"),
            ("not_list", "TypeError"),
            ("missing_deque", "IndexError"),
        ):
            self.assertIn(source, reasons[1])
            self.assertIn(exception_type, reasons[1])

        manager.no_tensor_aliasing_sources = [
            ConstantSource("len(1)"),
            LocalSource("duplicate_a"),
        ]
        with self.assertRaisesRegex(TypeError, "has no len"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager, scope
            )

        manager.no_tensor_aliasing_sources = [
            ConstantSource("1 / 0"),
            LocalSource("duplicate_a"),
        ]
        with self.assertRaisesRegex(ZeroDivisionError, "division by zero"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager, scope
            )

    def test_aliasing_guard_recompile_reason_with_mixed_source_errors(self):
        class TypeErrorDescriptor:
            @property
            def tensor(self):
                raise TypeError("descriptor failure")

            def __call__(self):
                raise TypeError("call failure")

        class AttributeErrorDescriptor:
            @property
            def tensor(self):
                raise AttributeError("attribute descriptor failure")

        class KeyErrorContainer:
            def __getitem__(self, index):
                raise KeyError("custom getitem failure")

        class AttributeErrorGetattr:
            def __getattr__(self, name):
                raise AttributeError("custom getattr failure")

        class DefaultsAttributeError:
            @property
            def __defaults__(self):
                raise AttributeError("defaults descriptor failure")

        class DefaultsTypeError(list):
            @property
            def __defaults__(self):
                raise TypeError("defaults descriptor type failure")

        class MissingDict(dict):
            def __missing__(self, key):
                raise KeyError("custom missing failure")

        class DictSubclass(dict):
            def __init__(self, stored, returned):
                super().__init__(key=stored)
                self.returned = returned

            def __getitem__(self, key):
                return self.returned

        class Holder:
            def __init__(self, data):
                self.data = data

        class ChangingHolder:
            def __init__(self, *values):
                self.values = iter(values)

            @property
            def data(self):
                return next(self.values)

        class HashErrorKey:
            fail = False

            def __hash__(self):
                if self.fail:
                    raise TypeError("key hash failure")
                return 0

        class TypeErrorHolder:
            @property
            def data(self):
                raise TypeError("index descriptor failure")

        manager = torch._dynamo.guards.GuardManagerWrapper()
        manager.global_scope = {}
        other = torch.randn(4)
        scope = {"L": {"items": [TypeErrorDescriptor()], "other": other}}

        for getitem_source in (
            GetItemSource(LocalSource("items"), 0),
            ListGetItemSource(LocalSource("items"), 0),
        ):
            for source, error in (
                (AttrSource(getitem_source, "tensor"), "descriptor failure"),
                (CallFunctionNoArgsSource(getitem_source), "call failure"),
            ):
                with self.subTest(source=source.name):
                    manager.no_tensor_aliasing_sources = [source, LocalSource("other")]
                    with self.assertRaisesRegex(TypeError, error):
                        torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                            manager, scope
                        )

        manager.no_tensor_aliasing_sources = [
            AttrSource(ListGetItemSource(LocalSource("attribute_items"), 0), "tensor"),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(AttributeError, "attribute descriptor failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {
                    "L": {
                        "attribute_items": [AttributeErrorDescriptor()],
                        "other": other,
                    }
                },
            )

        manager.no_tensor_aliasing_sources = [
            GetItemSource(LocalSource("custom_items"), 0),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(KeyError, "custom getitem failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {"L": {"custom_items": KeyErrorContainer(), "other": other}},
            )

        manager.no_tensor_aliasing_sources = [
            AttrSource(LocalSource("custom_attr"), "tensor"),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(AttributeError, "custom getattr failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {"L": {"custom_attr": AttributeErrorGetattr(), "other": other}},
            )

        manager.no_tensor_aliasing_sources = [
            DefaultsSource(LocalSource("custom_defaults"), 0),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(AttributeError, "defaults descriptor failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {
                    "L": {
                        "custom_defaults": DefaultsAttributeError(),
                        "other": other,
                    }
                },
            )

        manager.no_tensor_aliasing_sources = [
            DefaultsSource(LocalSource("custom_defaults"), 0),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(TypeError, "defaults descriptor type failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {
                    "L": {
                        "custom_defaults": DefaultsTypeError(),
                        "other": other,
                    }
                },
            )

        manager.no_tensor_aliasing_sources = [
            DictGetItemSource(LocalSource("custom_mapping"), "missing"),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(KeyError, "custom missing failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {"L": {"custom_mapping": MissingDict(), "other": other}},
            )

        overridden = torch.randn(4)
        manager.no_tensor_aliasing_sources = [
            DictSubclassGetItemSource(LocalSource("mapping"), "key"),
            LocalSource("overridden"),
        ]
        reasons = (
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {
                    "L": {
                        "mapping": DictSubclass(torch.randn(4), overridden),
                        "overridden": overridden,
                    }
                },
            )
        )
        self.assertIn("Duplicate tensors found", reasons[0])

        key = object()
        holder = ChangingHolder({key: torch.randn(4)}, {object(): torch.randn(4)})
        dict_source = AttrSource(LocalSource("holder"), "data")
        manager.no_tensor_aliasing_sources = [
            DictGetItemSource(dict_source, ConstDictKeySource(dict_source, 0)),
            LocalSource("other"),
        ]
        reasons = (
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager, {"L": {"holder": holder, "other": torch.randn(4)}}
            )
        )
        self.assertEqual(reasons, ["NO_TENSOR_ALIASING guard failed"])

        hash_error_key = HashErrorKey()
        mapping = {hash_error_key: torch.randn(4)}
        hash_error_key.fail = True
        mapping_source = LocalSource("mapping")
        manager.no_tensor_aliasing_sources = [
            DictGetItemSource(mapping_source, ConstDictKeySource(mapping_source, 0)),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(TypeError, "key hash failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {"L": {"mapping": mapping, "other": other}},
            )

        duplicate = torch.randn(4)
        holder = Holder([duplicate, duplicate])
        shared_source = AttrSource(LocalSource("holder"), "data")
        manager.no_tensor_aliasing_sources = [
            GetItemSource(shared_source, 0),
            GetItemSource(shared_source, 1),
        ]
        reasons = (
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager, {"L": {"holder": holder}}
            )
        )
        self.assertIn("Duplicate tensors found", reasons[0])

        index_source = AttrSource(LocalSource("holder"), "data")
        manager.no_tensor_aliasing_sources = [
            DictGetItemSource(
                LocalSource("mapping"), ConstDictKeySource(index_source, 0)
            ),
            LocalSource("other"),
        ]
        with self.assertRaisesRegex(TypeError, "index descriptor failure"):
            torch._dynamo.guards.recompilation_reason_for_no_tensor_aliasing_guard(
                manager,
                {
                    "L": {
                        "mapping": {key: torch.randn(4)},
                        "holder": TypeErrorHolder(),
                        "other": other,
                    }
                },
            )

    def test_object_alias_relation_guards_without_lambda(self):
        class Box:
            pass

        def foo(box_a, box_b, t):
            entries = {box_a, box_b}
            if len(entries) == 1:
                return t + 1
            return t - 1

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.tensor(0)

        with dc.patch(use_lamba_guard_for_object_aliasing=False):
            compiled = torch.compile(foo, backend=cnt, fullgraph=True)

            shared = Box()
            res_alias = compiled(shared, shared, x)
            self.assertEqual(res_alias.item(), 1)

            res_unique = compiled(Box(), Box(), x)
            self.assertEqual(res_unique.item(), -1)
            self.assertEqual(cnt.frame_count, 2)

        torch._dynamo.reset()

    def test_aliasing_guard_failures_with_globals(self):
        g1 = torch.randn([3])
        g2 = torch.randn([3])

        def foo(a):
            a.add_(g1)
            return g2 + 1

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_foo = torch.compile(foo, backend=cnt, fullgraph=True)

        z = torch.randn([3])
        cmp_result = compiled_foo(z.detach().clone())
        eager_result = foo(z.detach().clone())
        self.assertEqual(cmp_result, eager_result)
        self.assertEqual(cnt.frame_count, 1)

        g1 = g1.detach().clone()
        cmp_result = compiled_foo(g1)
        g1 = g1.detach().clone()
        eager_result = compiled_foo(g1)
        self.assertEqual(cmp_result, eager_result)
        # Recompile, alias changed
        self.assertEqual(cnt.frame_count, 2)

    def test_dynamic_shape_parameter_recompile(self):
        # Test the matrix multiplication with Parameters.
        # Without the config assume_parameters_shapes_static_by_default,
        # the torch.nn.Parameter shapes are assumed to be static which leads to recompilation

        w = torch.nn.Parameter(torch.randn(3, 2))

        def foo(x):
            return x @ w

        def run_foo_6_times_and_count_recompiles():
            cnt = torch._dynamo.testing.CompileCounter()

            opt = torch.compile(foo, backend=cnt, fullgraph=True)

            x = torch.nn.Parameter(torch.randn(1, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(10, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(11, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(15, 3))
            opt(x)
            x = torch.nn.Parameter(torch.randn(15, 3))
            opt(x)

            return cnt

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_default_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", True)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_default_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_static_comp_dynamic_param():
            return run_foo_6_times_and_count_recompiles()

        @patch.object(torch._dynamo.config, "force_parameter_static_shapes", False)
        @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", True)
        @patch.object(torch._dynamo.config, "assume_static_by_default", True)
        def run_dynamic_comp_dynamic_param():
            return run_foo_6_times_and_count_recompiles()

        torch._dynamo.reset()
        static_comp_default_param = run_static_comp_default_param()
        self.assertEqual(static_comp_default_param.frame_count, 4)
        self.assertEqual(static_comp_default_param.op_count, 4)

        torch._dynamo.reset()
        dynamic_comp_default_param = run_dynamic_comp_default_param()
        self.assertEqual(dynamic_comp_default_param.frame_count, 4)
        self.assertEqual(dynamic_comp_default_param.op_count, 4)

        torch._dynamo.reset()
        static_comp_dynamic_param = run_static_comp_dynamic_param()
        self.assertEqual(static_comp_dynamic_param.frame_count, 4)
        self.assertEqual(static_comp_dynamic_param.op_count, 4)

        torch._dynamo.reset()
        dynamic_comp_dynamic_param = run_dynamic_comp_dynamic_param()
        self.assertEqual(dynamic_comp_dynamic_param.frame_count, 2)
        self.assertEqual(dynamic_comp_dynamic_param.op_count, 2)

    def test_simple_module_recompile(self):
        class SimpleDropout(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.dropout(self.linear(x))

        model = SimpleDropout()
        x = torch.randn(10)
        counter = torch._dynamo.testing.CompileCounter()
        model = torch.compile(model, backend=counter, fullgraph=True)
        for _ in range(20):
            model.eval()
            model(x)
            model.train()
            model(x)
        self.assertEqual(counter.frame_count, 2)

    @patch.object(torch._dynamo.config, "recompile_limit", 2)
    def test_no_recursive_compile_after_cache_limit_hit(self):
        def f(x, n):
            x = x + n
            return g(x, n)

        def g(x, n):
            x = x + n
            return h(x, n)

        def h(x, n):
            return x + n

        counter = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=counter, dynamic=False)
        for i in range(10):
            opt_f(torch.ones(3), i)
        self.assertEqual(counter.frame_count, 2)

    def test_automatic_dynamic_on_closed_ints(self):
        def f(x):
            def g(y):
                return y + x

            return g

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def h(x, g):
            return g(x)

        for i in range(10):
            h(torch.randn(5), f(i))
        self.assertEqual(counter.frame_count, 2)

    @patch.object(torch._dynamo.config, "recompile_limit", 2)
    def test_run_mode_after_cache_limit_hit(self):
        def f(x, n):
            x = x + n
            if torch._dynamo.is_compiling():
                x = x + 1
            return g(x, n)

        def g(x, n):
            x = x + n
            if torch._dynamo.is_compiling():
                x = x + 2
            return x

        counter = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=counter, dynamic=False)
        # compiles
        self.assertEqual(opt_f(torch.ones(3), 0), torch.ones(3) + 3)
        self.assertEqual(opt_f(torch.ones(3), 1), torch.ones(3) + 5)
        # cache limit hit
        self.assertEqual(opt_f(torch.ones(3), 2), torch.ones(3) + 4)
        self.assertEqual(opt_f(torch.ones(3), 3), torch.ones(3) + 6)
        # run mode
        self.assertEqual(opt_f(torch.ones(3), 0), torch.ones(3) + 3)
        self.assertEqual(opt_f(torch.ones(3), 1), torch.ones(3) + 5)
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes_mark_as="unbacked")
    def test_automatic_dynamic_shapes_mark_as_unbacked(self):
        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x):
            return x * x

        f(torch.randn(3))
        f(torch.randn(2))
        f(torch.randn(1))
        f(torch.randn(0))

        self.assertEqual(counter.frame_count, 2)  # not three or four!

    @torch._dynamo.config.patch(recompile_limit=2, fail_on_recompile_limit_hit=True)
    def test_tensorify_python_builtin_mul_does_not_recompile(self):
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        def scaling_step(update, dummy_tensor, lr):
            dummy_tensor.mul_(0.5 * lr)

            r, c = update.size(-2), update.size(-1)
            scaling_factor = max(1, r / c) ** 0.5
            update.mul_(scaling_factor * lr)

            return update

        compiled = torch.compile(scaling_step, backend=counter, fullgraph=True)
        base_update = torch.randn(128, 128)
        base_dummy = torch.randn(324, 64)

        for i in range(8):
            lr = 1e-4 * (i + 1)
            self.assertEqual(
                compiled(base_update.clone(), base_dummy.clone(), lr),
                scaling_step(base_update.clone(), base_dummy.clone(), lr),
            )

        self.assertLessEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=2, fail_on_recompile_limit_hit=True)
    def test_tensorify_python_builtin_pow_does_not_recompile(self):
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        def step_param(v_t, bc2):
            eps = 1e-8
            return v_t.sqrt().div_(bc2**0.5).add_(eps)

        compiled = torch.compile(step_param, backend=counter, fullgraph=True)
        base_v_t = torch.randn(64, 1280)

        for step in range(1, 9):
            bc2 = 1.0 - 0.999**step
            self.assertEqual(
                compiled(base_v_t.clone(), bc2),
                step_param(base_v_t.clone(), bc2),
            )

        self.assertLessEqual(counter.frame_count, 2)

    def test_ambient_autocast_recompile(self):
        weights = torch.randn(10, 10)
        counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            return torch.mm(x, weights)

        x = torch.randn(1, 10)

        self.assertEqual(fn(x).dtype, torch.float32)

        with torch.autocast("cpu", torch.float16):
            self.assertEqual(fn(x).dtype, torch.float16)

        with torch.autocast("cpu", torch.bfloat16):
            self.assertEqual(fn(x).dtype, torch.bfloat16)

        # should recompile each time
        self.assertEqual(counter.frame_count, 3)

    def test_autocast_constant_fold(self):
        # test that constant-folded autocast functions
        # work properly - it should work if the global autocast
        # state is guarded.

        weights = torch.randn(10, 10)
        counter = torch._dynamo.testing.CompileCounterWithBackend("eager")

        def fn(x):
            if torch.get_autocast_dtype("cpu") == torch.float16:
                x = x + 1
            else:
                x = x - 1
            return torch.mm(x, weights)

        opt_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x = torch.randn(1, 10)

        with torch.autocast("cpu", torch.float16):
            self.assertEqual(fn(x), opt_fn(x))

        with torch.autocast("cpu", torch.bfloat16):
            self.assertEqual(fn(x), opt_fn(x))

        self.assertEqual(counter.frame_count, 2)

    def test_dunder_call_recompile(self):
        class Foo:
            def __call__(self, x):
                return x + 1

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter)
        def f(x, foo):
            return foo(x)

        x = torch.ones(2)
        foo1 = Foo()
        foo2 = Foo()

        # no recompilation
        f(x, foo1)
        f(x, foo2)
        self.assertEqual(counter.frame_count, 1)

        # one recompilation
        Foo.__call__ = lambda self, x: x + 2
        f(x, foo1)
        self.assertEqual(counter.frame_count, 2)

    def test_no_recompile_over_unused_objects(self):
        # This is a regression test case that imitates
        # https://github.com/city96/ComfyUI-GGUF/blob/47bec6147569a138dd30ad3e14f190a36a3be456/ops.py#L169-L182
        counter = torch._dynamo.testing.CompileCounter()

        def f(x, key, patches):
            return x * x + 1

        @torch.compile(backend=counter, fullgraph=True)
        def apply_patches(f, x, keys):
            patches = []
            for key, patch in keys:  # noqa: F402
                patches.append(patch)
            x = f(x, key, patches)
            return x

        # no recompilation
        x = torch.rand(10)
        apply_patches(f, x, [("a", 1), ("b", 2)])
        self.assertEqual(counter.frame_count, 1)
        apply_patches(f, x, [("c", 3), ("d", 4)])
        self.assertEqual(counter.frame_count, 1)

    def test_out_variant_does_not_overrecompile(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/135859.
        # The out= variants of max/min/topk used to recompile on every new input
        # shape because their out overloads (e.g. aten.max.dim_max) lacked a meta
        # function, so dynamic shapes were not propagated to the out tensors.
        # They should now recompile exactly once, like the functional variant.
        def count_recompiles(fn):
            cnt = torch._dynamo.testing.CompileCounter()
            opt = torch.compile(fn, backend=cnt, dynamic=None)
            for n in range(4, 10):
                opt(torch.randn(n, 8))
            return cnt.frame_count

        def max_out(x):
            values = x.new_empty(x.shape[0])
            indices = x.new_empty(x.shape[0], dtype=torch.long)
            torch.max(x, dim=1, out=(values, indices))
            return values, indices

        def min_out(x):
            values = x.new_empty(x.shape[0])
            indices = x.new_empty(x.shape[0], dtype=torch.long)
            torch.min(x, dim=1, out=(values, indices))
            return values, indices

        def topk_out(x):
            values = x.new_empty((x.shape[0], 3))
            indices = x.new_empty((x.shape[0], 3), dtype=torch.long)
            torch.topk(x, 3, dim=1, out=(values, indices))
            return values, indices

        for out_fn in (max_out, min_out, topk_out):
            torch._dynamo.reset()
            self.assertEqual(count_recompiles(out_fn), 2)

    @parametrize("mutation", ("code", "rebind"))
    @parametrize("inherited", (False, True))
    def test_class_function_attr_mutation_recompiles(self, inherited, mutation):
        # A function living on the class (not the instance __dict__) resolves to
        # a bound method. Both replacing its code object in place and rebinding
        # the class attribute change what eager runs, so both must invalidate
        # the compiled code. `inherited` moves the function off mro[0], which
        # sources it through type.__mro__[1].__dict__ instead.
        def two(self, x):
            return x * 2.0

        def nine(self, x):
            return x * 9.0

        class Base:
            m = two

        class Derived(Base):
            pass

        model = Derived() if inherited else Base()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(lambda x: model.m(x), backend=cnt, fullgraph=True)

        x = torch.arange(1.0, 4.0)
        self.assertEqual(opt_fn(x), x * 2.0)
        self.assertEqual(cnt.frame_count, 1)

        # Always mutate Base, the class the function actually lives on;
        # rebinding on Derived would instead be caught by the MRO shadowing
        # guards rather than by the code guard under test.
        if mutation == "code":
            Base.m.__code__ = nine.__code__
        else:
            Base.m = nine

        self.assertEqual(opt_fn(x), x * 9.0)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(opt_fn(x), x * 9.0)
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("access", ("instance", "class"))
    @parametrize("inherited", (False, True))
    def test_classmethod_code_mutation_recompiles(self, inherited, access):
        def two(cls, x):
            return x * 2.0

        def nine(cls, x):
            return x * 9.0

        class Base:
            c = classmethod(two)

        class Derived(Base):
            pass

        owner = Derived if inherited else Base
        cnt = torch._dynamo.testing.CompileCounter()
        if access == "instance":
            model = owner()
            opt_fn = torch.compile(lambda x: model.c(x), backend=cnt, fullgraph=True)
        else:
            opt_fn = torch.compile(lambda x: owner.c(x), backend=cnt, fullgraph=True)

        x = torch.arange(1.0, 4.0)
        self.assertEqual(opt_fn(x), x * 2.0)
        self.assertEqual(cnt.frame_count, 1)

        Base.c.__func__.__code__ = nine.__code__

        self.assertEqual(opt_fn(x), x * 9.0)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(opt_fn(x), x * 9.0)
        self.assertEqual(cnt.frame_count, 2)

    def test_unmutated_method_does_not_recompile(self):
        # The code guard must not fire on unrelated calls: a second instance of
        # the same class shares one code object and must hit the same entry.
        class Model:
            def m(self, x):
                return x * 2.0

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(obj, x):
            return obj.m(x)

        x = torch.arange(1.0, 4.0)
        fn(Model(), x)
        fn(Model(), x)
        self.assertEqual(cnt.frame_count, 1)


class FloatGuardBitwiseTests(torch._dynamo.test_case.TestCase):
    # Float constant guards must be value-identity (bitwise), not IEEE eq:
    # -0.0 == 0.0 so an EQUALS_MATCH guard built for 0.0 wrongly passed for
    # -0.0 and reused a graph with 0.0 baked in, while nan != nan needs
    # (and has) dedicated is-nan guards.

    def test_neg_zero_recompiles_and_is_correct(self):
        import math

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, dynamic=False)
        def f(x, s: float):
            return x * s, math.copysign(1.0, s)

        x = torch.randn(4)
        _, sign_pos = f(x, 0.0)
        _, sign_neg = f(x, -0.0)
        self.assertEqual(sign_pos, 1.0)
        self.assertEqual(sign_neg, -1.0)
        self.assertEqual(cnt.frame_count, 2)

    def test_nan_sign_recompiles_and_is_correct(self):
        import math
        import struct

        def from_bits(bits):
            return struct.unpack(">d", struct.pack(">Q", bits))[0]

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, dynamic=False)
        def f(x, s: float):
            return x + 1, math.copysign(1.0, s)

        # nan sign and payload are observable, so the nan guard must be
        # bitwise: a graph specialized on a positive nan must not be reused
        # for a negative nan.
        x = torch.randn(4)
        _, sign_pos = f(x, from_bits(0x7FF8000000000001))
        _, sign_neg = f(x, from_bits(0xFFF8000000001234))
        self.assertEqual(sign_pos, 1.0)
        self.assertEqual(sign_neg, -1.0)
        self.assertEqual(cnt.frame_count, 2)

    def test_nan_float_does_not_recompile(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, dynamic=False)
        def f(x, s: float):
            return x * s

        x = torch.randn(4)
        f(x, float("nan"))
        f(x, float("nan"))
        self.assertEqual(cnt.frame_count, 1)

    def test_complex_nan_guard_checks_other_component(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, dynamic=False)
        def f(x, c: complex):
            return x + c

        # The COMPLEX_IS_NAN guard must include the non-nan component:
        # complex(7, nan) must not reuse the graph specialized on
        # complex(5, nan), but the same constant must not recompile.
        x = torch.randn(4, dtype=torch.cfloat)
        f(x, complex(5.0, float("nan")))
        f(x, complex(7.0, float("nan")))
        self.assertEqual(cnt.frame_count, 2)
        f(x, complex(7.0, float("nan")))
        self.assertEqual(cnt.frame_count, 2)


instantiate_parametrized_tests(RecompileTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
