import torch
from torch.fx import symbolic_trace
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import run_tests


class TestASTRewriter(JitTestCase):

    def test_assert_no_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b
                return a + b
        m = M()
        traced = symbolic_trace(m)
        traced.graph.lint(traced)

    def test_assert_with_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, "test message"
                return a + b
        m = M()
        traced = symbolic_trace(m)
        traced.graph.lint(traced)

    def test_assert_with_empty_msg(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, ""
                return a + b
        m = M()
        traced = symbolic_trace(m)
        traced.graph.lint(traced)

    def test_assert_with_multiline_message(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                error_msg = """
An error message with
terrible spacing
                """
                assert a == b, error_msg
                return a + b
        m = M()
        traced = symbolic_trace(m)
        traced.graph.lint(traced)


if __name__ == '__main__':
    run_tests()
