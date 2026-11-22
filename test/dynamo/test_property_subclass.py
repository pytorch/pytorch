"""Regression tests for TorchDynamo property subclass support.

Covers the cases tracked in https://github.com/pytorch/pytorch/issues/195901:
  - Constructing property subclasses inside torch.compile
  - Slotted property subclasses with/without __doc__ slot
  - .getter() / .setter() / .deleter() chaining on subclasses
  - __doc__ read/write semantics matching CPython
  - Instance attribute access on property subclasses with __dict__
"""

import sys
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case


# ---------------------------------------------------------------------------
# Module-level class definitions (match style of test/cpython/v3_13/test_property.py)
# ---------------------------------------------------------------------------

class PropertySub(property):
    """This is a subclass of property"""


class PropertySubSlots(property):
    """Slotted subclass -- no __doc__ slot, so doc is dropped."""
    __slots__ = ()


class PropertySubclassTests(torch._dynamo.test_case.TestCase):
    """Regression tests for GitHub issue #195901 (property subclass in Dynamo)."""

    # ------------------------------------------------------------------
    # test_issue41287: subclass __doc__ and explicit doc= kwarg
    # ------------------------------------------------------------------
    @unittest.skipIf(sys.flags.optimize >= 2, "Docstrings are omitted with -O2 and above")
    def test_issue41287(self):
        """property subclass class docstring and doc= kwarg are preserved."""

        def fn():
            cls_doc = PropertySub.__doc__
            inst_doc = PropertySub(None, None, None, "issue 41287 is fixed").__doc__
            return cls_doc, inst_doc

        cls_doc, inst_doc = fn()
        self.assertEqual(cls_doc, "This is a subclass of property",
                         "Docstring of `property` subclass is ignored")
        self.assertEqual(inst_doc, "issue 41287 is fixed",
                         "Subclasses of `property` ignores `doc` constructor argument")

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        cls_doc2, inst_doc2 = opt_fn()
        self.assertEqual(cls_doc2, "This is a subclass of property")
        self.assertEqual(inst_doc2, "issue 41287 is fixed")

    # ------------------------------------------------------------------
    # Slotted subclass: no __doc__ slot -- doc silently dropped
    # ------------------------------------------------------------------
    def test_property_with_slots_no_docstring(self):
        """Slotted property subclass (no __doc__ slot): __doc__ reads None."""
        class slotted_prop(property):
            __slots__ = ("foo",)

        def fn():
            p = slotted_prop(lambda self: 1)
            return getattr(p, "__doc__", None)

        self.assertIsNone(fn())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertIsNone(opt_fn())

    @unittest.skipIf(sys.flags.optimize >= 2, "Docstrings are omitted with -O2 and above")
    def test_property_with_slots_docstring_silently_dropped(self):
        """Explicit doc= kwarg on slotted subclass without __doc__ slot: dropped."""
        class slotted_prop(property):
            __slots__ = ("foo",)

        def fn():
            p = slotted_prop(doc="what's up")
            return p.__doc__

        self.assertIsNone(fn())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertIsNone(opt_fn())

    # ------------------------------------------------------------------
    # Slotted subclass WITH __doc__ slot: doc is preserved
    # ------------------------------------------------------------------
    @unittest.skipIf(sys.flags.optimize >= 2, "Docstrings are omitted with -O2 and above")
    def test_property_with_slots_and_doc_slot_docstring_present(self):
        """Slotted subclass with __doc__ slot: doc kwarg and getter doc preserved."""
        class slotted_prop(property):
            __slots__ = ("foo", "__doc__")

        def fn_kwarg():
            p = slotted_prop(doc="what's up")
            return p.__doc__

        self.assertEqual(fn_kwarg(), "what's up")
        opt_fn = torch.compile(fn_kwarg, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "what's up")

        def documented_getter():
            """what's up getter doc?"""
            return 4

        def fn_getter():
            p = slotted_prop(documented_getter)
            return p.__doc__

        self.assertEqual(fn_getter(), "what's up getter doc?")
        opt_fn2 = torch.compile(fn_getter, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn2(), "what's up getter doc?")

    # ------------------------------------------------------------------
    # .getter() / .setter() / .deleter() chaining preserves subclass type
    # ------------------------------------------------------------------
    def test_property_subclass_getter_chaining(self):
        """property.getter() on a subclass returns the same subclass type."""

        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)
            p2 = p.getter(lambda self: 2)
            return type(p2).__name__, p2.fget(None)

        name, val = fn()
        self.assertEqual(name, "Sub")
        self.assertEqual(val, 2)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        name2, val2 = opt_fn()
        self.assertEqual(name2, "Sub")
        self.assertEqual(val2, 2)

    def test_property_subclass_setter_chaining(self):
        """property.setter() on a subclass preserves the subclass type."""

        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)
            p2 = p.setter(lambda self, v: None)
            return type(p2).__name__

        self.assertEqual(fn(), "Sub")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "Sub")

    def test_property_subclass_deleter_chaining(self):
        """property.deleter() on a subclass preserves the subclass type."""

        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)
            p2 = p.deleter(lambda self: None)
            return type(p2).__name__

        self.assertEqual(fn(), "Sub")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "Sub")

    # ------------------------------------------------------------------
    # Instance dict attribute access on subclasses with __dict__
    # ------------------------------------------------------------------
    def test_property_subclass_instance_dict_attr(self):
        """Property subclasses with __dict__ support arbitrary attribute sets."""

        class Tagged(property):
            pass

        def fn():
            p = Tagged(lambda self: 99)
            p.tag = "mytag"
            return p.tag

        self.assertEqual(fn(), "mytag")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "mytag")

    # ------------------------------------------------------------------
    # __doc__ write semantics
    # ------------------------------------------------------------------
    def test_property_doc_write_base(self):
        """property (base) __doc__ is writable."""

        def fn():
            p = property(lambda self: 1)
            p.__doc__ = "new doc"
            return p.__doc__

        self.assertEqual(fn(), "new doc")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "new doc")

    def test_property_subclass_doc_write_with_dict(self):
        """property subclass with __dict__: __doc__ is writable."""

        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)
            p.__doc__ = "written"
            return p.__doc__

        self.assertEqual(fn(), "written")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "written")

    def test_property_subclass_doc_write_readonly(self):
        """Slotted subclass WITHOUT __doc__ slot: writing __doc__ raises AttributeError."""

        class slotted_prop(property):
            __slots__ = ("foo",)

        def fn():
            p = slotted_prop(lambda self: 1)
            p.__doc__ = "nope"

        # Eager: AttributeError is raised directly.
        with self.assertRaises(AttributeError):
            fn()

        # Compiled: Dynamo surfaces it as Unsupported (observed exception at the
        # top-level compiled frame with no handler) rather than re-raising the
        # AttributeError directly.  Both confirm the write is rejected.
        import torch._dynamo.exc

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises((AttributeError, torch._dynamo.exc.Unsupported)):
            opt_fn()

    # ------------------------------------------------------------------
    # property used as a class-level descriptor (integration smoke test)
    # ------------------------------------------------------------------
    def test_property_subclass_as_descriptor(self):
        """property subclass installed on a class works under torch.compile."""

        class LoggedProperty(property):
            pass

        class MyObj:
            def __init__(self, v):
                self._v = v

            @LoggedProperty
            def value(self):
                return self._v

        def fn(obj):
            return obj.value

        obj = MyObj(7)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(obj), 7)


if __name__ == "__main__":
    unittest.main()
