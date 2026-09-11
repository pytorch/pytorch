import torch
from torch.testing._internal.common_utils import TestCase


class TestOptimizedModuleBool(TestCase):
    def test_bool_without_len(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        compiled = torch.compile(Model(), backend="eager")
        self.assertTrue(bool(compiled))

    def test_debugger_fix_installation_is_idempotent(self):
        from torch._dynamo import _bytecode_debugger_fix
        from torch._dynamo import bytecode_debugger, eval_frame

        debug_init = bytecode_debugger._DebugContext.__init__
        optimized_bool = eval_frame.OptimizedModule.__bool__
        _bytecode_debugger_fix.install()
        self.assertIs(bytecode_debugger._DebugContext.__init__, debug_init)
        self.assertIs(eval_frame.OptimizedModule.__bool__, optimized_bool)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
