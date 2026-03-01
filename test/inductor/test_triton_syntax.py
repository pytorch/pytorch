# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


class TestTritonSyntacticallyValid(TestCase):
    @requires_gpu()
    def test_triton_sqrt(self):
        # https://github.com/pytorch/pytorch/issues/142328
        import math

        import torch.nn as nn

        def newtonschulz5(G, steps: int, eps=1e-7):
            assert len(G.shape) == 2  # noqa: S101
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.to(
                torch.bfloat16
                if torch.cuda.is_bf16_supported(including_emulation=False)
                else torch.float16
            )
            X /= X.norm() + eps  # ensure top singular value <= 1
            if G.size(0) > G.size(1):
                X = X.T
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X
            if G.size(0) > G.size(1):
                X = X.T
            return X

        @torch.compile(backend="inductor")
        def scaled_newton_schulz(G, steps: int):
            shape = G.shape
            dtype = G.dtype
            G = G.reshape(shape[0], -1)
            G = newtonschulz5(G, steps)
            G = G.reshape(shape).type(dtype)
            G = G * math.sqrt(max(1, shape[0] / G[0].numel()))
            return G

        model = nn.Sequential(
            nn.Linear(16, 16, bias=False),
            nn.Linear(16, 32, bias=False),
        ).to(device=torch.device(GPU_TYPE))

        loss = model(torch.randn(4, 16, device=torch.device(GPU_TYPE))).sum()
        loss.backward()

        scaled_newton_schulz(model[0].weight.grad, 6)
        scaled_newton_schulz(model[1].weight.grad, 6)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
