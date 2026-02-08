# Owner(s): ["oncall: pt2"]

"""
Test suite for OpInfo ops with unbacked symints.

This test marks tensor dimensions as unbacked and verifies that ops
can be compiled with fullgraph=True without data-dependent errors (DDEs).
"""

import copy
import unittest

import torch
import torch._dynamo
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_ops_unbacked import ops_dde_xfail, ops_unbacked_skip
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TestCase
from torch.utils._pytree import tree_flatten, tree_map_


DEVICE_TYPE = "cpu"


def apply_skip_decorators(all_opinfos, test_case_name, base_test_name, to_skip):
    # Build lookup dict for O(n) performance
    opinfo_by_name = {}
    for o in all_opinfos:
        key = (o.name, o.variant_test_name)
        opinfo_by_name.setdefault(key, []).append(o)

    for xfail_entry in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail_entry
        matching_opinfos = opinfo_by_name.get((op_name, variant_name), [])
        # Some ops may not exist in op_db, skip silently
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(
                    unittest.expectedFailure,
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(
                    unittest.skip("Skipped!"),
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)


apply_skip_decorators(
    op_db, "TestOpsUnbacked", "test_unbacked_op_db", ops_dde_xfail | ops_unbacked_skip
)


class TestOpsUnbacked(TestCase):
    def _has_valid_unbacked_dims(self, t: torch.Tensor) -> bool:
        """Check if tensor has dimensions that can be marked as unbacked."""
        return t.ndim > 0 and any(s >= 2 for s in t.shape)

    def _mark_unbacked(self, t: torch.Tensor) -> None:
        """Mark all eligible dimensions of a tensor as unbacked."""
        for i in range(t.ndim):
            if t.shape[i] >= 2:
                torch._dynamo.decorators.mark_unbacked(t, i)

    def _run_with_unbacked_compile(self, func, args, kwargs):
        """
        Mark tensor dims as unbacked and run with fullgraph compile.
        Raises if a DDE occurs.
        """
        torch._dynamo.reset()

        def mark_unbacked_tree(x):
            if isinstance(x, torch.Tensor) and self._has_valid_unbacked_dims(x):
                self._mark_unbacked(x)
            return x

        tree_map_(mark_unbacked_tree, (args, kwargs))

        @torch.compile(backend="eager", fullgraph=True)
        def compiled_func(*a, **kw):
            return func(*a, **kw)

        compiled_func(*args, **kwargs)

    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_unbacked_op_db(self, device, dtype, op):
        samples = list(op.sample_inputs(device, dtype, requires_grad=False))

        any_tested = False

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            all_tensors = [
                x
                for x in tree_flatten((args, kwargs))[0]
                if isinstance(x, torch.Tensor)
            ]
            if not any(self._has_valid_unbacked_dims(t) for t in all_tensors):
                continue

            # # First verify the sample passes in eager mode
            op.op(*copy.deepcopy(args), **copy.deepcopy(kwargs))

            any_tested = True
            args_copy, kwargs_copy = copy.deepcopy((args, kwargs))
            self._run_with_unbacked_compile(op.op, args_copy, kwargs_copy)

        if not any_tested:
            self.fail("Should have skipped; no valid samples found")


instantiate_device_type_tests(TestOpsUnbacked, globals(), only_for=(DEVICE_TYPE,))

if __name__ == "__main__":
    run_tests()
