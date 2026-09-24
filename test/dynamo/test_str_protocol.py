# Owner(s): ["module: dynamo"]
"""Tests for tp_str / generic_str behavior in Dynamo."""

import collections
import logging
import sys
import typing
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
    subtest,
)


class _OpaqueStrDescriptorObject:
    __str__ = str.upper


class TpStrTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @make_dynamo_test
    def test_str_int(self):
        assert str(42) == "42"  # noqa: S101
        assert str(-1) == "-1"  # noqa: S101
        assert str(0) == "0"  # noqa: S101

    @make_dynamo_test
    def test_str_float(self):
        assert str(3.14) == "3.14"  # noqa: S101
        assert str(0.0) == "0.0"  # noqa: S101
        assert str(-2.5) == "-2.5"  # noqa: S101

    @make_dynamo_test
    def test_str_bool(self):
        assert str(True) == "True"  # noqa: S101
        assert str(False) == "False"  # noqa: S101

    @make_dynamo_test
    def test_str_none(self):
        assert str(None) == "None"  # noqa: S101

    @make_dynamo_test
    def test_str_string_identity(self):
        s = "hello"
        empty = ""
        assert str(s) == "hello"  # noqa: S101
        assert str(empty) == ""  # noqa: S101

    @make_dynamo_test
    def test_str_dunder_constant(self):
        assert (42).__str__() == "42"  # noqa: S101
        assert (3.14).__str__() == "3.14"  # noqa: S101
        assert True.__str__() == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_unbound_dunder_constant(self):
        assert int.__str__(42) == "42"  # noqa: S101
        assert float.__str__(3.14) == "3.14"  # noqa: S101
        assert bool.__str__(True) == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_unbound_dunder_string(self):
        assert str.__str__("hello") == "hello"  # noqa: S101
        assert str.__str__("") == ""  # noqa: S101

    @make_dynamo_test
    def test_str_list_falls_back_to_repr(self):
        assert str([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_object_dunder_str_on_string_uses_repr(self):
        assert object.__str__("hello") == "'hello'"  # noqa: S101
        assert object.__str__("") == "''"  # noqa: S101

    @make_dynamo_test
    def test_object_dunder_str_on_list_uses_repr(self):
        assert object.__str__([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101


class TpStrUserDefinedTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_counter_str(self):
        def fn(x):
            return str(collections.Counter("aba"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_defined_str(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return str(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_dunder_str(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return obj.__str__()

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_default_object_str(self):
        class Plain:
            pass

        def fn(x, obj):
            return str(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_repr_fallback_for_str(self):
        class MyObj:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            return str(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_str_on_plain_instance(self):
        class Plain:
            pass

        def fn(x, obj):
            return object.__str__(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_str_ignores_user_defined_str(self):
        class MyObj:
            def __str__(self):
                return "MyObj"

        def fn(x, obj):
            return object.__str__(obj)

        obj = MyObj()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, repr(obj))
        self.assertNotEqual(out, str(obj))

    def test_object_dunder_str_uses_user_defined_repr(self):
        class MyObj:
            def __repr__(self):
                return "MyObjRepr"

            def __str__(self):
                return "MyObjStr"

        def fn(x, obj):
            return object.__str__(obj)

        obj = MyObj()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, repr(obj))
        self.assertNotEqual(out, str(obj))

    def test_str_returning_non_string_raises(self):
        class BadStr:
            def __str__(self):
                return 3  # noqa: PLE0307

        def fn(x, obj):
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        obj = BadStr()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertIn("__str__", out)
        self.assertEqual(fn(x, obj), out)

    @unittest.expectedFailure
    def test_user_defined_opaque_str_descriptor_raises_type_error(self):
        def fn(x, obj):
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        eager_result = fn(x, _OpaqueStrDescriptorObject())
        self.assertIn(
            "descriptor 'upper' for 'str' objects doesn't apply",
            eager_result,
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(eager_result, compiled(x, _OpaqueStrDescriptorObject()))

    def test_metaclass_str(self):
        class Meta(type):
            def __repr__(cls):
                return f"<MetaRepr {cls.__name__}>"

            def __str__(cls):
                return f"<MetaStr {cls.__name__}>"

        class MyClass(metaclass=Meta):
            pass

        def fn(x):
            return str(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_type_dunder_str_on_class(self):
        class Meta(type):
            def __repr__(cls):
                return f"<MetaRepr {cls.__name__}>"

            def __str__(cls):
                return f"<MetaStr {cls.__name__}>"

        class MyClass(metaclass=Meta):
            pass

        def fn(x):
            return type.__str__(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))
        self.assertEqual(compiled(x), type.__str__(MyClass))

    def test_user_function_str(self):
        def helper(y):
            return y + 1

        def fn(x):
            return str(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_lambda_str(self):
        helper = lambda: None  # noqa: E731

        def fn(x):
            return str(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_str(self):
        def fn(x):
            return str(collections.defaultdict(int, {"a": 1}))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_str_with_nested_function_factory_unsupported(self):
        def fn(x):
            def factory():
                return x

            return str(collections.defaultdict(factory, {"a": 1}))

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_ordereddict_and_namedtuple_str_track_nested_repr(self):
        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        class Named(typing.NamedTuple):
            obj: object

        def fn(x, obj):
            ordered = collections.OrderedDict([("obj", obj)])
            named = Named(obj)
            y = x + 1
            s1 = (str(ordered), str(named))
            obj.val.append(0)
            s2 = (str(ordered), str(named))
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_structseq_str_with_tensor_graph_breaks(self):
        def fn(x):
            return str(torch.max(x, dim=0))

        x = torch.randn(3, 2)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), str(torch.max(x, dim=0)))


class TpStrExceptionTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @make_dynamo_test
    def test_exception_no_args(self):
        assert str(ValueError()) == ""  # noqa: S101

    @make_dynamo_test
    def test_exception_one_arg(self):
        assert str(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_exception_one_int_arg(self):
        assert str(ValueError(42)) == "42"  # noqa: S101

    @make_dynamo_test
    def test_exception_multiple_args(self):
        assert str(ValueError("error", 42)) == "('error', 42)"  # noqa: S101

    @make_dynamo_test
    def test_exception_dunder(self):
        assert TypeError("bad type").__str__() == "bad type"  # noqa: S101

    @make_dynamo_test
    def test_exception_unbound_dunder(self):
        assert ValueError.__str__(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_runtime_error(self):
        assert str(RuntimeError("runtime failure")) == "runtime failure"  # noqa: S101

    def test_user_defined_exception_subclass_str(self):
        class MyError(ValueError):
            pass

        def fn(x):
            return str(MyError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_defined_exception_subclass_custom_str(self):
        class MyError(ValueError):
            def __str__(self):
                return f"MyError({self.args[0]!r})"

        def fn(x):
            return str(MyError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))


@instantiate_parametrized_tests
class FStringGraphBreakTests(TestCase):
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_logger_fstring_debug_resumes_after_graph_break(self):
        test_logger = logging.getLogger(
            "test_logger_fstring_debug_resumes_after_graph_break"
        )

        def f(x):
            a = x + 1
            test_logger.warning(f"a : {a=}")  # noqa: G004
            b = x * 2
            return b + 1

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.ones(2)
        opt_f = torch.compile(f, backend=cnt)
        with self.assertLogs(test_logger, level="WARNING") as captured:
            opt_out = opt_f(x)
            second_out = opt_f(x + 1)

        self.assertEqual(opt_out, x * 2 + 1)
        self.assertEqual(second_out, (x + 1) * 2 + 1)
        self.assertEqual(
            [record.getMessage() for record in captured.records],
            ["a : a=tensor([2., 2.])", "a : a=tensor([3., 3.])"],
        )
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 3)

        graph_breaks = counters["graph_break"]
        repr_breaks = sum(
            count
            for reason, count in graph_breaks.items()
            if reason.startswith("repr() on tensor")
        )
        logging_breaks = sum(
            count
            for reason, count in graph_breaks.items()
            if reason.startswith("logging.Logger method not supported")
        )
        self.assertEqual(repr_breaks, 1)
        self.assertEqual(logging_breaks, 2)
        self.assertEqual(sum(graph_breaks.values()), repr_breaks + logging_breaks)

    @unittest.skipIf(sys.version_info < (3, 13), "requires split f-string opcodes")
    @parametrize("with_spec", (False, True))
    def test_split_format_opcode_resumes_unsupported(self, with_spec):
        def fail_format(*args, **kwargs):
            raise Unsupported("test f-string formatting graph break")

        def f(x):
            y = x + 1
            if with_spec:
                f"{x:}"
            else:
                f"{x}"
            return y + 1

        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.ones(2)
        with patch(
            "torch._dynamo.symbolic_convert.InstructionTranslatorBase._format_value",
            side_effect=fail_format,
        ):
            opt_out = torch.compile(f, backend=cnt)(x)

        self.assertEqual(opt_out, x + 2)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(sum(counters["graph_break"].values()), 1)


@instantiate_parametrized_tests
class FStringMutationTests(TestCase):
    """Tests for f-string mutation ordering (issue #177582).

    Dynamo must evaluate f-string formatting at the correct bytecode point
    so that mutations between two f-strings are reflected in the output.
    """

    hw_classification = HardwareClassification.GENERIC

    def _check(self, fn, *args_factory):
        import copy

        import torch
        import torch._dynamo.testing

        eager_result = fn(*copy.deepcopy(args_factory))
        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt)
        compiled_result = compiled_fn(*copy.deepcopy(args_factory))
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(cnt.frame_count, 1)

    def test_fstring_tracks_user_defined_object_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"obj = {obj}"
            obj.val.append(0)
            s2 = f"obj = {obj}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_fstring_tracks_frozen_dataclass_field_mutations(self):
        from dataclasses import dataclass

        import torch

        @dataclass(frozen=True)
        class FrozenObj:
            val: list

            def __repr__(self):
                return f"FrozenObj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"obj = {obj}"
            obj.val.append(0)
            s2 = f"obj = {obj}"
            return x, s1, s2

        self._check(fn, torch.randn(3), FrozenObj([1, 2]))

    def test_fstring_str_conversion_tracks_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"{obj!s}"
            obj.val.append(0)
            s2 = f"{obj!s}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_fstring_repr_conversion_tracks_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = f"{obj!r}"
            obj.val.append(0)
            s2 = f"{obj!r}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    @parametrize(
        "format_kind",
        (
            # TODO: Preserve formatting order without breaking fullgraph
            # reorderable logging, which relies on deferred tensor formatting.
            subtest("plain", decorators=[unittest.expectedFailure]),
            subtest("plain_with_spec", decorators=[unittest.expectedFailure]),
            "str",
            "repr",
            "ascii",
            "repr_with_spec",
        ),
    )
    def test_fstring_tensor_formatting_tracks_mutations(self, format_kind):
        def fn(x):
            x = x + 1
            if format_kind == "plain":
                s = f"{x}"
            elif format_kind == "plain_with_spec":
                s = f"{x:>20}"
            elif format_kind == "str":
                s = f"{x!s}"
            elif format_kind == "repr":
                s = f"{x!r}"
            elif format_kind == "ascii":
                s = f"{x!a}"
            elif format_kind == "repr_with_spec":
                s = f"{x!r:>20}"
            else:
                raise AssertionError(f"unexpected format kind: {format_kind}")
            x.add_(1)
            return s, x

        inp = (
            torch.tensor(1.0)
            if format_kind == "plain_with_spec"
            else torch.tensor([1.0, 2.0])
        )
        eager_result = fn(inp.clone())
        cnt = torch._dynamo.testing.CompileCounter()
        compiled_result = torch.compile(fn, backend=cnt)(inp.clone())
        self.assertEqual(compiled_result, eager_result)

        if format_kind not in ("plain", "plain_with_spec"):
            self.assertEqual(cnt.frame_count, 2)
            self.assertEqual(cnt.op_count, 2)
            self.assertEqual(sum(counters["graph_break"].values()), 1)

    def test_explicit_str_tracks_mutations(self):
        import torch

        class Obj:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return f"Obj({self.val})"

        def fn(x, obj):
            x = x + 1
            s1 = str(obj)
            obj.val.append(0)
            s2 = str(obj)
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))


if __name__ == "__main__":
    run_tests()
