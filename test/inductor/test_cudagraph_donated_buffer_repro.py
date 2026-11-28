# SPDX-License-Identifier: BSD-3-Clause
import unittest
import torch
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class TestCudagraphDonatedBuffer(TestCase):

    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires cuda and triton")
    def test_legacy_cudagraph_donated_buffer_first_backward(self):
        torch._dynamo.reset()

        model = (
            torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1),
            )
            .cuda()
        )

        x = torch.randn(1024, 512, device="cuda")
        y = (torch.rand(1024, device="cuda") < 0.3).float()

        # Compute eager reference gradients
        model.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(
            model(x).squeeze(-1), y
        ).backward()
        ref_grads = torch.cat([p.grad.flatten() for p in model.parameters()])

        # Test configuration triggering bug
        options = {"triton.cudagraphs": True, "triton.cudagraph_trees": False}

        with (
            torch._functorch.config.patch(donated_buffer=True),
            fresh_inductor_cache(),
        ):
            compiled = torch.compile(
                model, fullgraph=True, dynamic=False, options=options
            )

            for step in range(3):
                model.zero_grad(set_to_none=True)
                out = compiled(x).squeeze(-1)
                torch.nn.functional.binary_cross_entropy_with_logits(
                    out, y
                ).backward()
                comp_grads = torch.cat(
                    [p.grad.flatten() for p in model.parameters()]
                )

                rel_err = (
                    (comp_grads - ref_grads).norm() / ref_grads.norm()
                ).item()

                # Step 0 (first backward call) failed prior to the fix
                self.assertLess(
                    rel_err, 1e-3, f"Failed on backward step {step}"
                )


if __name__ == "__main__":
    run_tests()
