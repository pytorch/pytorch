# Owner(s): ["oncall: distributed"]

from unittest.mock import patch

import torch
from torch.distributed import _token_switch as token_switch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestTokenSwitchAllocations(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        original = torch.Tensor.new_empty

        def allocate(tensor, *shape, **kwargs):
            output = original(tensor, *shape, **kwargs)
            output.fill_(float("nan"))
            return output

        dirty_empty = patch.object(torch.Tensor, "new_empty", allocate)
        dirty_empty.start()
        self.addCleanup(dirty_empty.stop)

    @parametrize("layout", ["flat", "expert_major"])
    @parametrize("initialize_tokens", [True, False])
    def test_receive_token_initialization_preserves_metadata(
        self, layout, initialize_tokens
    ):
        tokens = torch.ones(4, 512, dtype=torch.bfloat16)
        scales = torch.ones(4, 16, dtype=torch.uint8)
        weights = torch.ones(4, 2, dtype=torch.float32)
        received, scores, ids, metadata = token_switch._dispatch_quantized_async_fake(
            None,
            tokens,
            scales,
            weights,
            16,
            layout,
            initialize_tokens=initialize_tokens,
        )
        self.assertEqual(received.shape, (16, 512))
        if initialize_tokens:
            self.assertEqual(torch.count_nonzero(received), 0)
        else:
            self.assertTrue(torch.isnan(received).all())
        self.assertEqual(torch.count_nonzero(scores), 0)
        self.assertEqual(torch.count_nonzero(metadata), 0)
        if layout == "flat":
            self.assertEqual(ids, torch.full((16, 2), -1, dtype=torch.int64))
        else:
            self.assertIsNone(ids)

    @parametrize("initialize_tokens", [True, False])
    def test_backward_receive_token_initialization(self, initialize_tokens):
        tokens = torch.ones(4, 512, dtype=torch.bfloat16)
        output = token_switch._dispatch_backward_async_fake(
            None, tokens, 16, initialize_tokens=initialize_tokens
        )
        self.assertEqual(output.shape, (16, 512))
        if initialize_tokens:
            self.assertEqual(torch.count_nonzero(output), 0)
        else:
            self.assertTrue(torch.isnan(output).all())

    def test_complete_combine_token_outputs_skip_initialization(self):
        tokens = torch.ones(16, 512, dtype=torch.bfloat16)
        scores = torch.ones(16, dtype=torch.float32)
        output = token_switch._combine_async_fake(None, tokens, 4, 16)
        token_grads, score_grads = token_switch._combine_backward_async_fake(
            None, tokens, scores, 4, 2
        )
        self.assertEqual(output.shape, (4, 512))
        self.assertEqual(token_grads.shape, (4, 512))
        self.assertTrue(torch.isnan(output).all())
        self.assertTrue(torch.isnan(token_grads).all())
        self.assertEqual(score_grads.shape, (4, 2))
        self.assertEqual(torch.count_nonzero(score_grads), 0)

    @parametrize("name", ["dispatch_quantized_async", "dispatch_backward_async"])
    def test_receive_initialization_schema_defaults_to_safe_padding(self, name):
        schema = getattr(torch.ops._nccl_ep, name).default._schema
        argument = schema.arguments[-1]
        self.assertEqual(argument.name, "initialize_tokens")
        self.assertIs(argument.default_value, True)
        self.assertTrue(argument.kwarg_only)


if __name__ == "__main__":
    run_tests()
