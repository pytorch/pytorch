# Owner(s): ["module: dynamo"]

import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class PropertyAttributeMutationTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("attr", ["fget", "fset", "fdel", "__isabstractmethod__"])
    @parametrize("delete", [False, True])
    @parametrize("route", ["builtin", "dunder", "descriptor"])
    def test_readonly_attributes(self, attr, delete, route):
        def fn(p):
            try:
                if route == "descriptor":
                    descriptor = property.__dict__[attr]
                    if delete:
                        descriptor.__delete__(p)
                    else:
                        descriptor.__set__(p, 42)
                elif route == "dunder":
                    if delete:
                        p.__delattr__(attr)
                    else:
                        p.__setattr__(attr, 42)
                elif delete:
                    delattr(p, attr)
                else:
                    setattr(p, attr, 42)
            except AttributeError as error:
                return str(error)
            return "not raised"

        p = property(lambda obj: 1)
        expected = fn(p)
        self.assertNotEqual(expected, "not raised")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(p), expected)
        self.assertEqual(compiled(p), expected)

    @parametrize("route", ["builtin", "dunder", "descriptor"])
    @parametrize("value", ["changed", 42, None])
    def test_doc_write_persists(self, route, value):
        attr = "__doc__"

        def fn(p):
            alias = p
            old_doc = p.__doc__
            if route == "descriptor":
                result = property.__dict__[attr].__set__(p, value)
            elif route == "dunder":
                result = p.__setattr__(attr, value)
            else:
                result = setattr(p, attr, value)
            return old_doc, alias.__doc__, result, alias

        p = property(lambda obj: 1, doc="original")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual = compiled(p)
        self.assertEqual(actual[:3], ("original", value, None))
        self.assertIs(actual[3], p)
        self.assertEqual(p.__doc__, value)
        self.assertEqual(compiled(p)[:3], (value, value, None))

    @parametrize("route", ["builtin", "dunder", "descriptor"])
    def test_doc_delete(self, route):
        def fn(p):
            if route == "descriptor":
                result = property.__dict__["__doc__"].__delete__(p)
            elif route == "dunder":
                result = p.__delattr__("__doc__")
            else:
                result = delattr(p, "__doc__")
            return p.__doc__, hasattr(p, "__doc__"), result

        p = property(lambda obj: 1, doc="original")
        actual = torch.compile(fn, backend="eager", fullgraph=True)(p)
        self.assertEqual(actual, (None, True, None))
        self.assertIsNone(p.__doc__)

    @unittest.skipIf(sys.version_info < (3, 13), "property.__name__ requires 3.13")
    def test_name_write_and_descriptor_read(self):
        def fn(p):
            p.__name__ = "updated"
            return property.__dict__["__name__"].__get__(p, property)

        p = property(lambda obj: 1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(p), "updated")
        self.assertEqual(p.__name__, "updated")

    @unittest.skipIf(sys.version_info < (3, 13), "property.__name__ requires 3.13")
    def test_name_none_is_present(self):
        def fn(p):
            return hasattr(p, "__name__"), getattr(p, "__name__", "missing")

        p = property(lambda obj: 1)
        p.__name__ = None
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(p), (True, None))

    @unittest.skipIf(sys.version_info >= (3, 13), "property.__name__ is writable")
    @parametrize("delete", [False, True])
    def test_name_mutation_before_313(self, delete):
        def fn(p):
            try:
                if delete:
                    del p.__name__
                else:
                    p.__name__ = "updated"
            except AttributeError as error:
                return str(error), hasattr(p, "__name__")
            return "not raised", True

        p = property(lambda obj: 1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(p), fn(p))
        self.assertFalse(hasattr(p, "__name__"))

    @parametrize("attr", ["__doc__", "__name__"])
    def test_cached_write_replays(self, attr):
        if attr == "__name__" and sys.version_info < (3, 13):
            self.skipTest("property.__name__ requires 3.13")

        def fn(x, p):
            setattr(p, attr, "updated")
            return x + 1, getattr(p, attr)

        p = property(lambda obj: 1)
        x = torch.ones(2)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for value in ("before first call", "before cached call"):
            setattr(p, attr, value)
            self.assertEqual(compiled(x, p), (x + 1, "updated"))
            self.assertEqual(getattr(p, attr), "updated")
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_local_property_doc_write(self):
        def fn():
            class C:
                @property
                def value(self):
                    return 1

            p = C.__dict__["value"]
            p.__doc__ = 42
            return p.__doc__

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(), 42)

    def test_doc_alias_and_graph_break(self):
        def fn(p, value):
            alias = p
            p.__doc__ = value
            torch._dynamo.graph_break()
            alias.__doc__.append(2)
            return p, p.__doc__

        p = property(lambda obj: 1)
        value = [1]
        actual = torch.compile(fn, backend="eager")(p, value)
        self.assertIs(actual[0], p)
        self.assertIs(actual[1], value)
        self.assertIs(p.__doc__, value)
        self.assertEqual(value, [1, 2])


instantiate_parametrized_tests(PropertyAttributeMutationTests)

if __name__ == "__main__":
    run_tests()
