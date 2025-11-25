# Owner(s): ["module: dynamo"]
"""
Test for TYPE_MATCH guard and ___check_type_id function.

This test demonstrates how the TYPE_MATCH guard works in PyTorch Dynamo.
When a function is compiled, Dynamo installs guards to ensure the compiled
code remains valid. TYPE_MATCH guards ensure that values maintain their
exact type (using type identity, not just type equality).
"""

import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
import re

from torch.testing._internal.common_utils import munge_exc


class TestCheckTypeId(torch._dynamo.test_case.TestCase):
    def test_type_match_with_different_values(self):
        """
        Test that TYPE_MATCH guard correctly identifies type mismatches.

        This test compiles a function that uses a global variable and verifies:
        1. The compiled function works with values of the same type
        2. The function recompiles when the type changes
        3. The ___check_type_id/check_obj_id guard is present in the generated code
        4. The check_type_id should present the user-friendly code that specify the type
        """
        counter = {"value": 0}

        # Define a global variable that we'll guard on
        class Config:
            multiplier = 2  # int type

        def fn(x):
            # This will trigger a TYPE_MATCH guard on Config.multiplier
            return x * Config.multiplier

        # Compile the function
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        # First call - should compile and install guards
        x = torch.randn(4)
        result1 = opt_fn(x)
        expected1 = x * 2
        self.assertTrue(torch.allclose(result1, expected1))

        # Get the cache entry to inspect guards
        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)

        # Check that the guard string contains check_type_id
        guard_str = str(cache_entries[0].guard_manager)
        self.assertIn("___check_obj_id", guard_str)
        self.assertIn("type=<class '__main__.TestCheckTypeId.test_type_match_with_different_values.<locals>.Config'>", guard_str)
        guard_str_id_annon = re.sub(r"\d{7,}", "<type_id>", munge_exc(guard_str), flags=re.MULTILINE)
        self.assertExpectedInline(guard_str_id_annon, """
TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:N in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:N in init_ambient_guards
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0), type=<class 'torch.Tensor'>, tag_safe=(True, False)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4], stride=[1])  # return x * Config.multiplier  # test/dynamo/test_check_type_id.py:N in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x * Config.multiplier  # test/dynamo/test_check_type_id.py:N in fn
| +- GuardManager: source=L['Config'], accessed_by=FrameLocalsGuardAccessor(key='Config', framelocals_idx=1), type=<class 'type'>, tag_safe=(True, False)
| | +- ID_MATCH: ___check_obj_id(L['Config'], <type_id>), type=<class '__main__.TestCheckTypeId.test_type_match_with_different_values.<locals>.Config'>  # return x * Config.multiplier  # test/dynamo/test_check_type_id.py:N in fn
""")


    def test_type_match_with_custom_classes(self):
        """
        Test TYPE_MATCH guard with custom class instances.

        Demonstrates that the guard checks type identity, not structural equality.
        """
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class Point2D:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        point = Point(1, 2)

        def fn(tensor):
            # Access point's attributes, triggering TYPE_MATCH guard on point
            return tensor + point.x + point.y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        # First call with Point instance
        x = torch.ones(4)
        result1 = opt_fn(x)
        expected1 = x + 1 + 2
        self.assertTrue(torch.allclose(result1, expected1))

        # Verify guard contains check_type_id
        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)
        guard_str = str(cache_entries[0].guard_manager)
        self.assertIn("___check_type_id", guard_str)
        self.assertIn("type=<class '__main__.TestCheckTypeId.test_type_match_with_custom_classes.<locals>.Point'>", guard_str)
        guard_str_id_annon = re.sub(r"\d{7,}", "<type_id>", munge_exc(guard_str), flags=re.MULTILINE)
        self.assertExpectedInline(guard_str_id_annon, """
TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:N in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:N in init_ambient_guards
| +- GuardManager: source=L['point'], accessed_by=FrameLocalsGuardAccessor(key='point', framelocals_idx=1), type=<class '__main__.TestCheckTypeId.test_type_match_with_custom_classes.<locals>.Point'>, tag_safe=(False, False)
| | +- TYPE_MATCH: ___check_type_id(L['point'], <type_id>), type=<class '__main__.TestCheckTypeId.test_type_match_with_custom_classes.<locals>.Point'>  # return tensor + point.x + point.y  # test/dynamo/test_check_type_id.py:N in fn
| | +- GuardManager: source=L['point'].x, accessed_by=GetAttrGuardAccessor(x), type=<class 'int'>, tag_safe=(True, False)
| | | +- EQUALS_MATCH: L['point'].x == 1                                             # return tensor + point.x + point.y  # test/dynamo/test_check_type_id.py:N in fn
| | +- GuardManager: source=L['point'].y, accessed_by=GetAttrGuardAccessor(y), type=<class 'int'>, tag_safe=(True, False)
| | | +- EQUALS_MATCH: L['point'].y == 2                                             # return tensor + point.x + point.y  # test/dynamo/test_check_type_id.py:N in fn
| +- GuardManager: source=L['tensor'], accessed_by=FrameLocalsGuardAccessor(key='tensor', framelocals_idx=0), type=<class 'torch.Tensor'>, tag_safe=(True, False)
| | +- TENSOR_MATCH: check_tensor(L['tensor'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4], stride=[1])  # return tensor + point.x + point.y  # test/dynamo/test_check_type_id.py:N in fn
| | +- NO_HASATTR: hasattr(L['tensor'], '_dynamo_dynamic_indices') == False      # return tensor + point.x + point.y  # test/dynamo/test_check_type_id.py:N in fn
""")



if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
