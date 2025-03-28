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

    def test_failed_unbacked_view(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f1(x, ys):
            u, v = ys.tolist()
            return x.view(u, v)

        x = torch.randn(6, 6)[1:5, 1:5]  # make non-contiguous
        y = torch.tensor([4, 4])
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"Could not view a tensor .*4, 4.* as .*u0, u1.*"
            r"as the tensor is not contiguous",
        ):
            f1(x, y)

        @torch.compile(fullgraph=True, backend="eager")
        def f2(x, y):
            n = y.item()
            return x.reshape(y)

        x = torch.randn(4)
        y = torch.tensor([4])
        torch._dynamo.decorators.mark_unbacked(x, 0)
        f2(x, y)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
