# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case


@torch._dynamo.config.patch("capture_scalar_outputs", True)
class ViewTests(torch._dynamo.test_case.TestCase):
    def test_view_to_2d(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _u0):
            u0 = t[0].item()
            u1 = t[1].item()
            torch._check_is_size(u0)
            torch._check_is_size(u1)
            n = u0 * u1
            a = torch.randn(n)
            return a.view(-1, _u0)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 2)

    def test_view_to_1d(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _n):
            u0 = t[0].item()
            u1 = t[1].item()
            torch._check_is_size(u0)
            torch._check_is_size(u1)
            a = torch.randn(u0, u1)
            return a.view(_n)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 8)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
