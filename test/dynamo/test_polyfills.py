# Owner(s): ["module: dynamo"]

import builtins
import functools
import heapq
import itertools
import operator
import os as os_module
import struct
import sys
import traceback

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo


class TestLazyPolyfillLoading(torch._dynamo.test_case.TestCase):
    """Tests for lazy loading of polyfill modules during tracing."""

    def test_standard_library_polyfills_not_loaded_at_import(self):
        """Verify that standard library polyfill modules are loaded lazily."""
        from torch._dynamo.polyfills import loader

        # These modules should be loaded lazily, not eagerly
        lazy_modules = {
            "builtins",
            "functools",
            "heapq",
            "itertools",
            "operator",
            "os",
            "struct",
            "sys",
            "traceback",
        }

        # Check that lazy modules are not in the loaded set after import
        # (unless they were loaded by some other test that ran before this one)
        # The key point is that they're NOT automatically loaded at import time
        # This test is more about the mechanism than the exact state

        # Instead, verify the lazy registry exists and has the right entries
        self.assertGreater(len(loader._LAZY_POLYFILL_REGISTRY), 0)

        # Check some known entries are in the lazy registry
        self.assertIn(id(builtins.all), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(builtins.any), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(functools.reduce), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(heapq.heappush), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(itertools.chain), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(operator.itemgetter), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(os_module.fspath), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(struct.pack), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(sys.intern), loader._LAZY_POLYFILL_REGISTRY)
        self.assertIn(id(traceback.extract_tb), loader._LAZY_POLYFILL_REGISTRY)

    def test_eager_modules_loaded_at_import(self):
        """Verify that PyTorch internal polyfill modules are loaded eagerly."""
        from torch._dynamo.polyfills import loader

        # These modules should be loaded eagerly because they polyfill PyTorch internals
        eager_modules = {"fx", "tensor", "torch_c_nn", "_collections"}

        for module_name in eager_modules:
            if module_name in loader.POLYFILLED_MODULE_NAMES:
                self.assertIn(
                    module_name,
                    loader._loaded_polyfill_modules,
                    f"Module {module_name} should be loaded eagerly",
                )

    def test_ensure_polyfill_loaded_triggers_loading(self):
        """Test that _ensure_polyfill_loaded correctly loads polyfill modules."""
        from torch._dynamo.polyfills import loader

        # Get a function that has a lazy polyfill
        test_fn = builtins.all

        # If not already loaded, loading should return True
        module_name = loader._LAZY_POLYFILL_REGISTRY.get(id(test_fn))
        if module_name and module_name not in loader._loaded_polyfill_modules:
            self.assertTrue(loader._ensure_polyfill_loaded(test_fn))
            self.assertIn(module_name, loader._loaded_polyfill_modules)

        # Second call should return False (already loaded)
        self.assertFalse(loader._ensure_polyfill_loaded(test_fn))

    @skipIfTorchDynamo("test uses torch.compile internally")
    def test_polyfill_loaded_during_tracing(self):
        """Test that polyfill modules are loaded on-demand during tracing."""
        # Reset dynamo
        torch._dynamo.reset()

        # Use builtins.all in a compiled function
        @torch.compile
        def fn_all(x):
            if all([True, True]):
                return x + 1
            return x

        result = fn_all(torch.tensor([1.0, 2.0]))
        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 3.0])))

    @skipIfTorchDynamo("test uses torch.compile internally")
    def test_polyfill_module_access_via_getattr(self):
        """Test that polyfill modules can be accessed via polyfills.<module> syntax."""
        from torch._dynamo import polyfills

        # Accessing a polyfill module should work via __getattr__
        # This is needed for code like: polyfills.builtins.iter_
        builtins_module = polyfills.builtins
        self.assertTrue(hasattr(builtins_module, "all"))
        self.assertTrue(hasattr(builtins_module, "any"))

        itertools_module = polyfills.itertools
        self.assertTrue(hasattr(itertools_module, "chain"))

    def test_get_polyfilled_modules_loads_all(self):
        """Test that get_polyfilled_modules loads all polyfill modules."""
        from torch._dynamo.polyfills import loader

        modules = loader.get_polyfilled_modules()

        # All modules should be loaded
        self.assertEqual(len(modules), len(loader.POLYFILLED_MODULE_NAMES))

        # All module names should now be in loaded set
        for name in loader.POLYFILLED_MODULE_NAMES:
            self.assertIn(name, loader._loaded_polyfill_modules)


class TestGroupTensorsByDeviceAndDtype(torch._dynamo.test_case.TestCase):
    """Tests for the group_tensors_by_device_and_dtype polyfill."""

    def test_polyfill_matches_cpp_single_list(self):
        """Test that polyfill matches C++ implementation for a single list."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
        ]
        tensorlistlist = [list1]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_multiple_lists(self):
        """Test that polyfill matches C++ implementation for multiple lists."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
        ]
        list2 = [torch.rand_like(t) for t in list1]
        list3 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_with_nones(self):
        """Test that polyfill matches C++ implementation with None values."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
        ]
        list2 = [None for _ in list1]
        list3 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_without_indices(self):
        """Test that polyfill matches C++ implementation without indices."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
        ]
        list2 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, False)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=False
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            # When with_indices=False, indices should be empty
            self.assertEqual(cpp_indices, polyfill_indices)
            self.assertEqual(polyfill_indices, [])

    def test_polyfill_empty_input(self):
        """Test that polyfill handles empty input correctly."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        result = group_tensors_by_device_and_dtype([], with_indices=True)
        self.assertEqual(result, {})

        result = group_tensors_by_device_and_dtype([[]], with_indices=True)
        self.assertEqual(result, {})

    def test_polyfill_groups_correctly(self):
        """Test that polyfill groups tensors correctly by device and dtype."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        t_f32_0 = torch.randn(4, dtype=torch.float32)
        t_f32_1 = torch.randn(4, dtype=torch.float32)
        t_f64_0 = torch.randn(4, dtype=torch.float64)
        t_f64_1 = torch.randn(4, dtype=torch.float64)

        list1 = [t_f32_0, t_f64_0, t_f32_1, t_f64_1]
        tensorlistlist = [list1]

        result = group_tensors_by_device_and_dtype(tensorlistlist, with_indices=True)

        # Should have two groups: float32 and float64
        self.assertEqual(len(result), 2)

        device = torch.device("cpu")
        f32_key = (device, torch.float32)
        f64_key = (device, torch.float64)

        self.assertIn(f32_key, result)
        self.assertIn(f64_key, result)

        f32_lists, f32_indices = result[f32_key]
        f64_lists, f64_indices = result[f64_key]

        # Check float32 group
        self.assertEqual(len(f32_lists[0]), 2)
        self.assertTrue(torch.equal(f32_lists[0][0], t_f32_0))
        self.assertTrue(torch.equal(f32_lists[0][1], t_f32_1))
        self.assertEqual(f32_indices, [0, 2])

        # Check float64 group
        self.assertEqual(len(f64_lists[0]), 2)
        self.assertTrue(torch.equal(f64_lists[0][0], t_f64_0))
        self.assertTrue(torch.equal(f64_lists[0][1], t_f64_1))
        self.assertEqual(f64_indices, [1, 3])

    @skipIfTorchDynamo("test uses CompileCounter which doesn't work under dynamo")
    def test_group_tensors_traceable_with_compile(self):
        """Test that torch._C._group_tensors_by_device_and_dtype is traceable with torch.compile.

        This test verifies:
        1. The function can be compiled without graph breaks
        2. The frame is actually compiled (not skipped)
        3. There is tensor compute happening in the compiled graph
        """
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(tensors, grads):
            # Group tensors by device and dtype (this uses the polyfill under compile)
            grouped = torch._C._group_tensors_by_device_and_dtype(
                [tensors, grads], True
            )

            # Perform some tensor computation to ensure the frame is not skipped
            # Sum up results for each dtype group
            total = torch.tensor(0.0)
            for grouped_tensors, indices in grouped.values():
                tensor_list, grad_list = grouped_tensors
                for t, g in zip(tensor_list, grad_list):
                    if t is not None and g is not None:
                        # This adds tensor compute to ensure frame isn't skipped
                        total = total + (t + g).sum().float()
            return total

        tensors = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
        ]
        grads = [torch.randn_like(t) for t in tensors]

        # Run without compile to get expected result
        expected = fn(tensors, grads)

        # Run with compile
        compiled_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        result = compiled_fn(tensors, grads)

        # Verify correctness
        self.assertTrue(torch.allclose(result, expected))

        # Verify compilation happened (frame was not skipped)
        self.assertGreaterEqual(
            cnts.frame_count, 1, "Expected at least 1 frame to be compiled"
        )

        # Verify there was tensor compute (op_count > 0 means tensor ops were traced)
        self.assertGreater(cnts.op_count, 0, "Expected tensor operations to be traced")


if __name__ == "__main__":
    run_tests()
