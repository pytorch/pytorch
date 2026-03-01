# Owner(s): ["module: inductor"]
# mypy: allow-untyped-defs
# flake8: noqa: TOR901
"""
Tests for symmetric memory argument registry.

This test suite validates the symm_mem argument registration system, which allows
operators to declare which arguments require symmetric memory allocation.
"""

import unittest

import torch
from torch._library.simple_registry import singleton, SymmMemArgsHolder
from torch.library import Library  # noqa: TOR901
from torch.testing._internal.common_utils import run_tests, TestCase


# Import to trigger symm_mem args registration
# This needs to happen at module level so registrations are available for tests
try:
    import torch.distributed._symmetric_memory  # noqa: F401
except ImportError:
    pass  # Module may not be available in all builds


def register_symm_mem_args(op, arg_names, validate=True):
    """Helper function for tests to register symm_mem args."""
    from torch._ops import OpOverload

    if isinstance(op, str):
        qualname = op
        op_overload = None
    elif isinstance(op, OpOverload):
        qualname = op.__qualname__
        op_overload = op if validate else None
    else:
        if hasattr(op, "__qualname__"):
            qualname = op.__qualname__
            op_overload = op if validate else None
        else:
            raise TypeError(f"Expected OpOverload or string, got {type(op)}")

    entry = singleton.find(qualname)
    entry.symm_mem_args.register(arg_names, op_overload=op_overload)


class TestSymmMemRegistry(TestCase):
    """Test suite for SymmMemArgsHolder core functionality."""

    def setUp(self):
        """Clear test entries from registry before each test."""
        super().setUp()
        # Only clear test entries, preserve real registrations (e.g., symm_mem::*)
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]

    def tearDown(self):
        """Clear test entries from registry after each test."""
        # Only clear test entries, preserve real registrations (e.g., symm_mem::*)
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]
        super().tearDown()

    def test_basic_registration_with_string(self):
        """Test basic registration using string qualname."""
        register_symm_mem_args(
            "test_namespace::test_op",
            ["input"],
            validate=False,
        )

        entry = singleton.find("test_namespace::test_op")
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

    def test_multiple_args_registration(self):
        """Test registering multiple arguments for an operator."""
        register_symm_mem_args(
            "test_namespace::test_op",
            ["input", "output", "buffer"],
            validate=False,
        )

        entry = singleton.find("test_namespace::test_op")
        args = entry.symm_mem_args.get()
        self.assertEqual(set(args), {"input", "output", "buffer"})
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("output"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("other"))

    def test_empty_arg_list_raises(self):
        """Test that registering empty arg list raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Cannot register empty arg_names"):
            register_symm_mem_args("test_namespace::test_op", [], validate=False)

    def test_unregistered_op_queries(self):
        """Test querying unregistered operators."""
        # Note: find() creates an entry, but symm_mem_args won't be registered
        entry = singleton.find("nonexistent::op")
        self.assertFalse(entry.symm_mem_args.is_registered())
        self.assertIsNone(entry.symm_mem_args.get())
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("input"))

    def test_registration_with_op_overload(self):
        """Test registration with actual OpOverload object."""
        op = torch.ops.aten.add.Tensor

        # Register without validation (since we're using arbitrary arg names)
        register_symm_mem_args(op, ["self"], validate=False)

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))

    def test_validation_with_valid_args(self):
        """Test that validation passes with valid argument names."""
        op = torch.ops.aten.add.Tensor

        # 'self' and 'other' are valid arguments for add.Tensor
        register_symm_mem_args(op, ["self", "other"], validate=True)

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("other"))

    def test_validation_with_invalid_args(self):
        """Test that validation fails with invalid argument names."""
        op = torch.ops.aten.add.Tensor

        # 'nonexistent_arg' is not a valid argument for add.Tensor
        with self.assertRaisesRegex(ValueError, "Invalid argument names"):
            register_symm_mem_args(op, ["nonexistent_arg"], validate=True)

    def test_overwrite_registration(self):
        """Test that re-registering an op overwrites previous registration."""
        qualname = "test_namespace::test_op"

        register_symm_mem_args(qualname, ["input"], validate=False)
        entry = singleton.find(qualname)
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

        # Overwrite with different args
        register_symm_mem_args(qualname, ["output"], validate=False)
        self.assertEqual(set(entry.symm_mem_args.get()), {"output"})

    def test_get_all_registered(self):
        """Test getting all registered operators."""
        register_symm_mem_args("ns1::op1", ["a"], validate=False)
        register_symm_mem_args("ns2::op2", ["b", "c"], validate=False)

        # Check that both are registered
        entry1 = singleton.find("ns1::op1")
        entry2 = singleton.find("ns2::op2")
        self.assertEqual(set(entry1.symm_mem_args.get()), {"a"})
        self.assertEqual(set(entry2.symm_mem_args.get()), {"b", "c"})

    def test_holder_isolation(self):
        """Test that creating a new SymmMemArgsHolder instance is isolated."""
        new_holder = SymmMemArgsHolder("test::op")

        # Register in singleton
        register_symm_mem_args("test::op", ["input"], validate=False)

        # New holder should be empty
        self.assertFalse(new_holder.is_registered())

        # But singleton should have it registered
        entry = singleton.find("test::op")
        self.assertTrue(entry.symm_mem_args.is_registered())


class TestLibraryIntegration(TestCase):
    """Test suite for Library.register_symm_mem_args integration."""

    def setUp(self):
        """Clear test entries from registry before each test."""
        super().setUp()
        # Only clear test entries, preserve real registrations (e.g., symm_mem::*)
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]

    def tearDown(self):
        """Clear test entries from registry after each test."""
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]
        super().tearDown()

    def test_library_def_registration(self):
        """Test registration via Library.DEF."""
        lib = Library("test_lib_def", "DEF")
        lib.define("test_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("test_op", ["input"])

        qualname = "test_lib_def::test_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    def test_library_fragment_registration(self):
        """Test registration via Library.FRAGMENT."""
        lib = Library("aten", "FRAGMENT")
        lib.register_symm_mem_args("add.Tensor", ["self"])

        qualname = "aten::add.Tensor"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))

    def test_library_multiple_args(self):
        """Test registering multiple args via Library."""
        lib = Library("test_lib_multi", "DEF")
        lib.define("test_op(Tensor a, Tensor b, Tensor out) -> Tensor")
        lib.register_symm_mem_args("test_op", ["a", "b", "out"])

        qualname = "test_lib_multi::test_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("a"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("b"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("out"))

    def test_library_with_op_overload(self):
        """Test Library registration with OpOverload object."""
        op = torch.ops.aten.mul.Tensor
        lib = Library("aten", "FRAGMENT")
        lib.register_symm_mem_args(op, ["self", "other"])

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("other"))


@unittest.skipIf(
    not torch.cuda.is_available() or not hasattr(torch.ops, "symm_mem"),
    "Requires CUDA and symm_mem ops",
)
class TestSymmMemOpsRegistration(TestCase):
    """Test suite for registering actual symm_mem operations."""

    def test_symm_mem_registrations_loaded(self):
        """Test that all symm_mem operations are properly registered."""
        try:
            import torch.distributed._symmetric_memory  # noqa: F401
        except ImportError:
            self.skipTest("torch.distributed._symmetric_memory not available")

        # Expected operations that should be registered
        expected_ops = [
            "symm_mem::one_shot_all_reduce",
            "symm_mem::one_shot_all_reduce_out",
            "symm_mem::one_shot_all_reduce_copy",
            "symm_mem::one_shot_all_reduce_copy_out",
            "symm_mem::two_shot_all_reduce_",
            "symm_mem::two_shot_all_reduce_out",
            "symm_mem::multimem_all_reduce_",
            "symm_mem::multimem_one_shot_all_reduce",
            "symm_mem::multimem_one_shot_all_reduce_out",
            "symm_mem::multimem_one_shot_reduce_out",
            "symm_mem::multimem_all_gather_out",
            "symm_mem::reduce_scatter_out",
            "symm_mem::all_to_all_vdev",
            "symm_mem::all_to_all_vdev_2d",
            "symm_mem::all_to_all_vdev_2d_offset",
            "symm_mem::tile_reduce",
            "symm_mem::multi_root_tile_reduce",
        ]

        for qualname in expected_ops:
            entry = singleton.find(qualname)
            self.assertTrue(
                entry.symm_mem_args.is_registered(),
                f"{qualname} does not have symm_mem_args registered",
            )

    def test_lowering_auto_registration(self):
        """Test that lowerings are auto-registered from registry metadata."""
        try:
            import torch.distributed._symmetric_memory  # noqa: F401
        except ImportError:
            self.skipTest("torch.distributed._symmetric_memory not available")

        try:
            _ = torch.ops.symm_mem.one_shot_all_reduce
        except AttributeError:
            self.skipTest("symm_mem ops not available")

        from torch._inductor import comm_lowering

        try:
            comm_lowering.register_symm_mem_lowerings()
        except Exception as e:
            self.fail(f"Lowering registration failed: {e}")

        symm_mem_count = sum(
            1
            for qualname, entry in singleton._data.items()
            if qualname.startswith("symm_mem::") and entry.symm_mem_args.is_registered()
        )
        self.assertGreater(
            symm_mem_count, 0, "Expected at least some symm_mem operations registered"
        )

    def test_custom_operator_with_symm_mem(self):
        """Test that custom operators can register and use symm_mem args."""
        lib = Library("test_custom_symm", "DEF")
        lib.define("my_collective(Tensor input, str group_name) -> Tensor")

        # Register symm_mem args
        lib.register_symm_mem_args("my_collective", ["input"])

        # Verify registration
        qualname = "test_custom_symm::my_collective"
        entry = singleton.find(qualname)

        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

        # Verify we can query the registered args
        symm_mem_args = entry.symm_mem_args.get()
        self.assertEqual(list(symm_mem_args), ["input"])


if __name__ == "__main__":
    run_tests()
