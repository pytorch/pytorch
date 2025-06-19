# Owner(s): ["module: inductor"]
import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_HELION, requires_helion


if HAS_HELION:
    import helion
    import helion.language as hl


class HelionTests(TestCase):
    @requires_helion()
    def test_add_kernel(self):
        @helion.kernel(config=helion.Config(block_sizes=[1, 2]))
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # match pytorch broadcasting rules
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty(
                x.shape,
                # match type promotion of torch.add
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            # tile will be a tuple of blocks
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return add(x, y)

        x = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)

        out = add(x, y)
        compiled_add = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(compiled_out, x + y)


instantiate_parametrized_tests(HelionTests)


if __name__ == "__main__":
    run_tests()
