# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_LINUX, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestGroupFusionFxPasses(TestCase):
    @torch._inductor.config.patch(group_fusion_fx_passes=True)
    @torch._inductor.config.patch(split_cat_fx_passes=True)
    def test_group_layer_norm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1_ws = [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
                self.l1_bs = [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
                self.l2_ws = [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
                self.l2_bs = [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]

            def forward(self, x):
                post_l1 = []
                for l1_out, l1_w, l1_b in zip(
                    torch.split(x, 10, dim=2), self.l1_ws, self.l1_bs
                ):
                    post_l1.append(
                        torch.nn.functional.layer_norm(
                            l1_out, (10,), weight=l1_w, bias=l1_b
                        )
                    )

                l1_out = torch.cat(post_l1, dim=2)

                post_l2 = []
                for l2_out, l2_w, l2_b in zip(
                    torch.split(l1_out, 10, dim=2), self.l2_ws, self.l2_bs
                ):
                    post_l2.append(
                        torch.nn.functional.layer_norm(
                            l2_out,
                            (
                                5,
                                10,
                            ),
                            weight=l2_w,
                            bias=l2_b,
                        )
                    )

                return torch.cat(post_l2, dim=2)

        args = [
            torch.randn(2, 5, 50),
        ]

        module = TestModule()

        expected = module(*args)
        actual = torch.compile(module, dynamic=True)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            counters["inductor"]["layer_norm_removed"],
            10,
        )
        self.assertEqual(
            counters["inductor"]["layer_norm_added"],
            2,
        )
        self.assertEqual(
            counters["inductor"]["scmerge_split_removed"],
            2,
        )
        self.assertEqual(
            counters["inductor"]["scmerge_cat_removed"],
            2,
        )
        counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA and not TEST_WITH_ROCM:
        run_tests()
