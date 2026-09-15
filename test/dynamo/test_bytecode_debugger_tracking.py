from torch.testing._internal.common_utils import TestCase

from torch._dynamo._bytecode_debugger_fix import _CodeIdSet


class TestBytecodeDebuggerTracking(TestCase):
    def test_code_objects_are_retained(self):
        tracked = _CodeIdSet()

        def first():
            return 1

        def second():
            return 2

        tracked.add(first.__code__)
        self.assertIn(first.__code__, tracked)
        self.assertNotIn(second.__code__, tracked)

        tracked.discard(first.__code__)
        self.assertNotIn(first.__code__, tracked)


if __name__ == "__main__":
    import torch

    torch.testing._internal.common_utils.run_tests()
