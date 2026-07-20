# Owner(s): ["module: inductor"]
# mypy: allow-untyped-defs
"""
Tests for symmetric memory argument registry.

This test suite validates the symm_mem argument registration system, which allows
operators to declare which arguments require symmetric memory allocation.
"""

import unittest
from unittest.mock import patch

import torch
from torch._library.simple_registry import singleton, SymmMemArgsHolder
from torch.library import Library  # noqa: SCOPED_LIBRARY
from torch.testing._internal.common_utils import run_tests, TestCase


def register_symm_mem_args(op, arg_names):
    """Helper function for tests to register symm_mem args."""
    from torch._ops import OpOverload

    if isinstance(op, str):
        qualname = op
    elif isinstance(op, OpOverload):
        qualname = op.__qualname__
    else:
        if hasattr(op, "__qualname__"):
            qualname = op.__qualname__
        else:
            raise TypeError(f"Expected OpOverload or string, got {type(op)}")

    entry = singleton.find(qualname)
    entry.symm_mem_args.register(arg_names)


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
        )

        entry = singleton.find("test_namespace::test_op")
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

    def test_multiple_args_registration(self):
        """Test registering multiple arguments for an operator."""
        register_symm_mem_args(
            "test_namespace::test_op",
            ["input", "output", "buffer"],
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
            register_symm_mem_args("test_namespace::test_op", [])

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
        register_symm_mem_args(op, ["self"])

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))

    def test_overwrite_registration(self):
        """Test that re-registering an op overwrites previous registration."""
        qualname = "test_namespace::test_op"

        register_symm_mem_args(qualname, ["input"])
        entry = singleton.find(qualname)
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

        # Overwrite with different args
        register_symm_mem_args(qualname, ["output"])
        self.assertEqual(set(entry.symm_mem_args.get()), {"output"})

    def test_get_all_registered(self):
        """Test getting all registered operators."""
        register_symm_mem_args("ns1::op1", ["a"])
        register_symm_mem_args("ns2::op2", ["b", "c"])

        # Check that both are registered
        entry1 = singleton.find("ns1::op1")
        entry2 = singleton.find("ns2::op2")
        self.assertEqual(set(entry1.symm_mem_args.get()), {"a"})
        self.assertEqual(set(entry2.symm_mem_args.get()), {"b", "c"})

    def test_holder_isolation(self):
        """Test that creating a new SymmMemArgsHolder instance is isolated."""
        new_holder = SymmMemArgsHolder("test::op")

        # Register in singleton
        register_symm_mem_args("test::op", ["input"])

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
        lib = Library("test_lib_def", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("test_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("test_op", ["input"])

        qualname = "test_lib_def::test_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    def test_library_fragment_registration(self):
        """Test registration via Library.FRAGMENT."""
        lib = Library("aten", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.register_symm_mem_args("add.Tensor", ["self"])

        qualname = "aten::add.Tensor"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))

    def test_library_multiple_args(self):
        """Test registering multiple args via Library."""
        lib = Library("test_lib_multi", "DEF")  # noqa: SCOPED_LIBRARY
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
        lib = Library("aten", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.register_symm_mem_args(op, ["self", "other"])

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("other"))

    def test_library_cleanup_on_delete(self):
        """Test that deleting a Library cleans up its symm_mem_args registration."""
        lib = Library("test_lib_cleanup", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input, str reduce_op, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        qualname = "test_lib_cleanup::my_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())

        del lib

        self.assertFalse(entry.symm_mem_args.is_registered())
        self.assertIsNone(entry.symm_mem_args.get())


class TestFunctionalOpCompile(TestCase):
    """Test that functional ops with registered symm_mem_args work with torch.compile."""

    def setUp(self):
        super().setUp()
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]

    def tearDown(self):
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]
        torch._dynamo.reset()
        super().tearDown()

    @unittest.skipIf(not torch.cuda.is_available(), "Requires CUDA")
    def test_functional_op_compiles_with_symm_mem_args(self):
        """Test that a functional op with registered symm_mem_args compiles and runs."""
        lib = Library("test_func_symm", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_functional_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_functional_op", ["input"])

        @torch.library.impl(lib, "my_functional_op", "Meta")
        def meta_impl(input, group_name):
            return torch.empty_like(input)

        @torch.library.impl(lib, "my_functional_op", "CUDA")
        def cuda_impl(input, group_name):
            return input + 1.0

        def f_eager(x):
            return torch.ops.test_func_symm.my_functional_op(x, "test_group")

        @torch.compile(backend="inductor", fullgraph=True)
        def f_compiled(x):
            return torch.ops.test_func_symm.my_functional_op(x, "test_group")

        x = torch.randn(4, 4, device="cuda")

        eager_result = f_eager(x.clone())
        compiled_result = f_compiled(x.clone())
        torch.testing.assert_close(compiled_result, eager_result)

        expected = x + 1.0
        torch.testing.assert_close(compiled_result, expected)

        # Verify the registration is visible in the registry
        entry = singleton.find("test_func_symm::my_functional_op")
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    @unittest.skipIf(not torch.cuda.is_available(), "Requires CUDA")
    def test_functional_op_with_multiple_symm_mem_args(self):
        """Test that multiple symm_mem args are registered and visible during compilation."""
        lib = Library("test_func_multi", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define(
            "my_multi_arg_op(Tensor input, Tensor out, str group_name) -> Tensor"
        )
        lib.register_symm_mem_args("my_multi_arg_op", ["input", "out"])

        @torch.library.impl(lib, "my_multi_arg_op", "Meta")
        def meta_impl(input, out, group_name):
            return torch.empty_like(input)

        @torch.library.impl(lib, "my_multi_arg_op", "CUDA")
        def cuda_impl(input, out, group_name):
            # use both symm_mem args to verify functionality
            return input + out

        def f_eager(x, y):
            return torch.ops.test_func_multi.my_multi_arg_op(x, y, "test_group")

        @torch.compile(backend="inductor", fullgraph=True)
        def f_compiled(x, y):
            return torch.ops.test_func_multi.my_multi_arg_op(x, y, "test_group")

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        eager_result = f_eager(x.clone(), y.clone())
        compiled_result = f_compiled(x.clone(), y.clone())
        torch.testing.assert_close(compiled_result, eager_result)

        expected = x + y
        torch.testing.assert_close(compiled_result, expected)

        # Verify both args are registered
        entry = singleton.find("test_func_multi::my_multi_arg_op")
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("out"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    def test_unregistered_op_skips_realization(self):
        """Test that _maybe_realize_symm_mem_args returns early for unregistered ops."""
        from torch._inductor.ir import FallbackKernel

        # Use an op unlikely to be registered by other tests
        op = torch.ops.aten.abs.default

        # Verify the registry has no symm_mem_args for this op
        entry = singleton.get(op.__qualname__)
        if entry is not None:
            self.assertFalse(entry.symm_mem_args.is_registered())

        # Should return immediately without error or side effects
        FallbackKernel._maybe_realize_symm_mem_args(op, torch.randn(2))

    def test_maybe_realize_finds_registered_op(self):
        """Test that _maybe_realize_symm_mem_args correctly looks up registered ops."""
        from torch._inductor.ir import FallbackKernel

        lib = Library("test_realize_lookup", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        op = torch.ops.test_realize_lookup.my_op.default

        # Verify the op is found in the registry
        entry = singleton._data.get(op.__qualname__)
        self.assertIsNotNone(entry)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

        # Call _maybe_realize_symm_mem_args directly. It should find the
        # registration but return early because group_name is a raw string
        # (not a graph-time value) and the tensor arg is a raw tensor
        # (not a TensorBox). No error expected.
        FallbackKernel._maybe_realize_symm_mem_args(op, torch.randn(4, 4), "test_group")

    def test_maybe_realize_skips_without_group_name(self):
        """Test that _maybe_realize_symm_mem_args returns early when group_name is absent."""
        from torch._inductor.ir import FallbackKernel

        lib = Library("test_no_group", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        op = torch.ops.test_no_group.my_op.default

        # No group_name argument → should return early without error
        FallbackKernel._maybe_realize_symm_mem_args(op, torch.randn(4, 4))

    def test_realize_called_for_tensorbox_with_registered_args(self):
        """Test that _maybe_realize_symm_mem_args calls realize_as_comm_buffer for TensorBox args."""
        from unittest.mock import MagicMock

        from torch._inductor.ir import FallbackKernel, TensorBox

        lib = Library("test_realize_call", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        op = torch.ops.test_realize_call.my_op.default
        mock_tbox = MagicMock(spec=TensorBox)

        realize_log = []

        def mock_realize(t, comm_type, group_name):
            realize_log.append((str(comm_type), group_name))

        with (
            patch(
                "torch._inductor.comm_lowering.can_realize_as_comm_buffer",
                return_value=True,
            ),
            patch(
                "torch._inductor.comm_lowering.realize_as_comm_buffer",
                side_effect=mock_realize,
            ),
        ):
            FallbackKernel._maybe_realize_symm_mem_args(op, mock_tbox, "test_group")

        self.assertEqual(len(realize_log), 1)
        self.assertIn("SYMM_MEM", realize_log[0][0])
        self.assertEqual(realize_log[0][1], "test_group")


if __name__ == "__main__":
    run_tests()
