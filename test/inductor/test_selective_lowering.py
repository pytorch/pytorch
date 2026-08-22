# Owner(s): ["module: inductor"]
"""
Test selective lowering control via node metadata annotations.
"""

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, parametrize
from torch.testing._internal.inductor_utils import patch_custom_fallback_pass
from torch.utils._triton import has_triton


class SelectiveLoweringTest(InductorTestCase):
    """
    Tests for user-controllable selective lowering using node.meta annotations.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("fallback", (True, False))
    def test_basic_selective_lowering(self, device, fallback: bool):
        """
        Test that nodes marked for fallback use fallback handlers instead of lowerings.
        """

        def foo(x, y):
            a = x + y  # This will be marked for fallback
            b = a * 2  # This will use normal lowering
            return b

        x = torch.randn(10, device=device)
        y = torch.randn(10, device=device)

        # Mark all add operations for fallback
        def should_fallback_add(node: torch.fx.Node) -> bool:
            return node.target == torch.ops.aten.add.Tensor if fallback else False

        with patch_custom_fallback_pass(should_fallback_add):
            compiled_fn = torch.compile(foo)
            result, (code,) = run_and_get_code(compiled_fn, x, y)

        # Check numerics.
        expected = foo(x, y)
        self.assertTrue(torch.allclose(result, expected))

        # Strip comments from the code.
        code = "\n".join(line for line in code.split("\n") if "#" not in line)

        # Check for the add fallback.
        self.assertEqual("aten.add" in code, fallback)


instantiate_device_type_tests(
    SelectiveLoweringTest, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if has_triton():
        run_tests(needs="filelock")
