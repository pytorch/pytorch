# Owner(s): ["module: dynamo"]

import importlib.util
import sys
import types
import unittest.mock as mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import device_interface
from torch._dynamo.device_interface import (
    _autocast_class_location,
    DeviceInterface,
    get_device_autocast_class_locations,
    get_device_autocast_classes,
    register_interface_for_device,
)
from torch._dynamo.variables.torch import (
    _matches_device_autocast_class,
    device_type_for_autocast_class,
)


STUB_DEVICE = "privateuseone"


class _StubAutocast(torch.amp.autocast_mode.autocast):
    """An out-of-tree backend's autocast with an implicit device_type."""

    def __init__(self, dtype=torch.float16, enabled=True, cache_enabled=None):
        super().__init__(
            STUB_DEVICE, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )


class _NarrowStubAutocast(torch.amp.autocast_mode.autocast):
    """An out-of-tree autocast that narrows __init__ to no arguments."""

    def __init__(self):
        super().__init__(STUB_DEVICE)


class _ReorderedStubAutocast(torch.amp.autocast_mode.autocast):
    """An out-of-tree autocast whose parameters differ from the base.

    Modelled on ``torch.npu.amp.autocast``, whose signature is
    ``(enabled=True, dtype=torch.float16, cache_enabled=True)`` -- a different
    order and different defaults than the base class.
    """

    def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
        super().__init__(
            STUB_DEVICE, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )


class _UnregisteredSiblingAutocast(torch.amp.autocast_mode.autocast):
    """Defined in this same file, but deliberately never registered.

    Guards the difference between keying on the defining file alone (which would
    match this) and keying on file plus qualified name (which does not).
    """

    def __init__(self):
        super().__init__(STUB_DEVICE)


class AutocastInterface(DeviceInterface):
    autocast_classes = frozenset(
        {_StubAutocast, _NarrowStubAutocast, _ReorderedStubAutocast}
    )


DUPLICATE_MODULE_NAME = "_dup_device_autocast"


class DeviceAutocastTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        # Register the built-in interfaces before snapshotting device_interfaces,
        # so restoring the snapshot cannot undo init_device_reg() for the rest
        # of the process.
        device_interface.get_registered_device_interfaces()
        # Cleanups run LIFO, so clearing the cache is registered first in order
        # to run after the snapshot is restored.
        self.addCleanup(get_device_autocast_classes.cache_clear)
        self.addCleanup(get_device_autocast_class_locations.cache_clear)
        patcher = mock.patch.dict(device_interface.device_interfaces)
        patcher.start()
        self.addCleanup(patcher.stop)
        get_device_autocast_classes.cache_clear()
        get_device_autocast_class_locations.cache_clear()
        self._register_stub_device_module()

    def _register_stub_device_module(self):
        """Give STUB_DEVICE an AMP-capable device module.

        ``torch.amp.autocast_mode.autocast.__init__`` requires the privateuse1
        backend to expose ``get_amp_supported_dtype()``.  When the backend has
        been renamed by a real out-of-tree backend, ``STUB_DEVICE`` is no longer
        that backend's name and the check does not apply.
        """
        if torch._C._get_privateuse1_backend_name() != STUB_DEVICE:
            return
        if hasattr(torch, STUB_DEVICE):
            return
        module = types.ModuleType(f"torch.{STUB_DEVICE}")
        module.get_amp_supported_dtype = lambda: [torch.float16]
        torch._register_device_module(STUB_DEVICE, module)

        def _unregister():
            delattr(torch, STUB_DEVICE)
            sys.modules.pop(f"torch.{STUB_DEVICE}", None)

        self.addCleanup(_unregister)

    def _reimport_this_file(self):
        """Load this file again under a second module name.

        An out-of-tree backend commonly ends up importable under two dotted names
        (``torch_foo.foo.amp.autocast_mode`` and ``torch.foo.amp.autocast_mode``),
        which loads two module objects from one file and runs the class body
        twice.  This reproduces that, including the ``sys.modules`` entry a real
        import would leave behind: the returned module's classes are *different*
        objects from this module's, defined in the same file.
        """
        spec = importlib.util.spec_from_file_location(DUPLICATE_MODULE_NAME, __file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[DUPLICATE_MODULE_NAME] = module
        self.addCleanup(sys.modules.pop, DUPLICATE_MODULE_NAME, None)
        spec.loader.exec_module(module)
        return module

    def _compile_autocast_region(self, autocast_class, *ctor_args):
        """Trace ``with autocast_class(*ctor_args):`` and return the enter nodes."""
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            with autocast_class(*ctor_args):
                return x + 1

        fn(torch.ones(2))
        self.assertEqual(len(graphs), 1)
        return [
            node
            for node in graphs[0].graph.nodes
            if node.op == "call_function" and node.target is torch.amp._enter_autocast
        ]

    def test_slot_defaults_to_empty(self):
        self.assertEqual(DeviceInterface.autocast_classes, frozenset())

    def test_matches_only_when_registered(self):
        self.assertFalse(_matches_device_autocast_class(_StubAutocast))

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertTrue(_matches_device_autocast_class(_StubAutocast))

    def test_device_type_derived_from_registration_key(self):
        self.assertIsNone(device_type_for_autocast_class(_StubAutocast))

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)

    def test_device_type_strips_index_from_key(self):
        register_interface_for_device(f"{STUB_DEVICE}:3", AutocastInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)

    def test_late_registration_invalidates_cache(self):
        get_device_autocast_classes()

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertIn(_StubAutocast, get_device_autocast_classes())

    def test_base_autocast_never_matches(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertFalse(
            _matches_device_autocast_class(torch.amp.autocast_mode.autocast)
        )

    def test_non_class_never_matches(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertFalse(_matches_device_autocast_class(torch.sin))
        self.assertIsNone(device_type_for_autocast_class(torch.sin))

    def test_unregistered_subclass_not_matched(self):
        """Matching is by exact registration, not by inheritance or location."""

        class _SiblingAutocast(_StubAutocast):
            pass

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertFalse(_matches_device_autocast_class(_SiblingAutocast))
        self.assertIsNone(device_type_for_autocast_class(_SiblingAutocast))

    def test_same_file_unregistered_class_not_matched(self):
        """A class sharing the defining file with a registered one must not match.

        This is what keying on (file, qualname) buys over keying on the file.
        """
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertEqual(
            _autocast_class_location(_UnregisteredSiblingAutocast)[0],
            _autocast_class_location(_StubAutocast)[0],
        )
        self.assertIsNone(device_type_for_autocast_class(_UnregisteredSiblingAutocast))

    def test_class_reimported_under_another_name_matches(self):
        """The registered class, reached through a second import of its file.

        Real out-of-tree backends hit this: the interface captures the class from
        one module object while traced code reaches the copy created by a second
        import, so identity alone silently stops matching.
        """
        register_interface_for_device(STUB_DEVICE, AutocastInterface)
        duplicate = self._reimport_this_file()._StubAutocast

        self.assertIsNot(duplicate, _StubAutocast)
        self.assertNotIn(duplicate, get_device_autocast_classes())
        self.assertTrue(_matches_device_autocast_class(duplicate))
        self.assertEqual(device_type_for_autocast_class(duplicate), STUB_DEVICE)

    def test_reimported_unregistered_class_not_matched(self):
        """The second import must not widen what matches."""
        register_interface_for_device(STUB_DEVICE, AutocastInterface)
        duplicate = self._reimport_this_file()._UnregisteredSiblingAutocast

        self.assertIsNot(duplicate, _UnregisteredSiblingAutocast)
        self.assertIsNone(device_type_for_autocast_class(duplicate))

    def test_class_without_a_source_file_matches_by_identity(self):
        """A registered class that has no file to key on still matches."""
        fileless = type(
            "_FilelessAutocast",
            (_StubAutocast,),
            {"__module__": "builtins"},  # builtins has no __file__
        )

        class FilelessInterface(DeviceInterface):
            autocast_classes = frozenset({fileless})

        self.assertIsNone(_autocast_class_location(fileless))
        register_interface_for_device(STUB_DEVICE, FilelessInterface)

        self.assertEqual(device_type_for_autocast_class(fileless), STUB_DEVICE)

    def test_reimported_class_traced(self):
        """End to end: the duplicate class still produces an autocast node."""
        register_interface_for_device(STUB_DEVICE, AutocastInterface)
        duplicate = self._reimport_this_file()._StubAutocast

        nodes = self._compile_autocast_region(duplicate)

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].args[0], STUB_DEVICE)

    def test_registered_autocast_class_traced(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        enters = self._compile_autocast_region(_StubAutocast)

        self.assertEqual(len(enters), 1)
        self.assertEqual(enters[0].args[0], STUB_DEVICE)
        self.assertEqual(enters[0].args[1], torch.float16)

    def test_narrowed_constructor_traced(self):
        """A subclass need not accept every base parameter."""
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        enters = self._compile_autocast_region(_NarrowStubAutocast)

        self.assertEqual(len(enters), 1)
        device_type, dtype, enabled, cache_enabled = enters[0].args
        self.assertEqual(device_type, STUB_DEVICE)
        # Parameters the subclass does not accept fall back to the defaults of
        # torch.amp.autocast_mode.autocast, which is what its super().__init__
        # call gets when it does not pass them either.
        self.assertIsNone(dtype)
        self.assertEqual(enabled, True)
        self.assertIsNone(cache_enabled)

    def test_subclass_defaults_are_used_not_base_defaults(self):
        """apply_defaults() must come from the subclass signature."""
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        enters = self._compile_autocast_region(_ReorderedStubAutocast)

        self.assertEqual(len(enters), 1)
        device_type, dtype, enabled, cache_enabled = enters[0].args
        self.assertEqual(device_type, STUB_DEVICE)
        self.assertEqual(dtype, torch.float16)
        self.assertEqual(enabled, True)
        self.assertEqual(cache_enabled, True)

    def test_positional_arg_binds_to_subclass_signature(self):
        """A subclass may order its parameters differently than the base."""
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        # First positional parameter is `enabled`, not the base's `dtype`.
        enters = self._compile_autocast_region(_ReorderedStubAutocast, False)

        self.assertEqual(len(enters), 1)
        device_type, dtype, enabled, cache_enabled = enters[0].args
        self.assertEqual(device_type, STUB_DEVICE)
        self.assertEqual(dtype, torch.float16)
        self.assertEqual(enabled, False)

    def test_internal_autocast_subclass_not_matched(self):
        from torch.amp.autocast_mode import _UnmanagedAutocast

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertFalse(_matches_device_autocast_class(_UnmanagedAutocast))

    def test_registration_rejects_non_autocast_class(self):
        class NotAutocastInterface(DeviceInterface):
            autocast_classes = frozenset({int})

        with self.assertRaisesRegex(TypeError, "is not a subclass"):
            register_interface_for_device(STUB_DEVICE, NotAutocastInterface)

    def test_registration_rejects_base_autocast(self):
        class BaseAutocastInterface(DeviceInterface):
            autocast_classes = frozenset({torch.amp.autocast_mode.autocast})

        with self.assertRaisesRegex(ValueError, "must not contain"):
            register_interface_for_device(STUB_DEVICE, BaseAutocastInterface)

    def test_conflicting_device_types_raise(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)
        register_interface_for_device("meta", AutocastInterface)

        with self.assertRaisesRegex(RuntimeError, "exactly one device_type"):
            get_device_autocast_classes()

    def test_indexed_keys_are_not_a_conflict(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)
        register_interface_for_device(f"{STUB_DEVICE}:0", AutocastInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
