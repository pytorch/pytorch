# Owner(s): ["oncall: distributed"]

from unittest.mock import Mock, patch

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
class TestTokenSwitchAutograd(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _routing(self):
        return token_switch.Routing(None, torch.zeros(2, 1, dtype=torch.int64))

    @parametrize("received_weights", [False, True])
    def test_base_token_only_backward(self, received_weights):
        routing = self._routing()
        tokens = torch.randn(2, 4, requires_grad=True)
        weights = torch.ones(2, 1, dtype=torch.bfloat16)
        switch = Mock(spec=token_switch.TokenSwitch)
        switch._alloc_dispatch_outputs.return_value = (
            torch.zeros_like(tokens),
            torch.zeros_like(weights) if received_weights else None,
            None,
        )

        def dispatch(routing, tokens, weights, output, scores, ids):
            output.copy_(tokens)
            if scores is not None:
                scores.copy_(weights)

        def combine(routing, expert_tokens, output):
            output.copy_(expert_tokens)

        switch._dispatch.side_effect = dispatch
        switch._combine.side_effect = combine
        received, _, _ = token_switch._DispatchAutograd.apply(
            switch, routing, tokens, weights, 2
        )
        received.square().sum().backward()
        self.assertEqual(tokens.grad, 2 * tokens.detach())
        switch._combine.assert_called_once()
        switch._dispatch_backward.assert_not_called()
        self.assertIsNone(weights.grad)

    def test_async_token_only_backward(self):
        routing = self._routing()
        tokens = torch.randn(2, 4, requires_grad=True)
        weights = torch.ones(2, 1, dtype=torch.bfloat16)
        native = Mock()

        def dispatch(handle, tokens, weights, output, scores, ids, *, async_op):
            output.copy_(tokens)
            scores.copy_(weights)
            ids.zero_()

        def combine(handle, expert_tokens, output, *, async_op):
            output.copy_(expert_tokens)

        native._nccl_ep_dispatch.side_effect = dispatch
        native._nccl_ep_combine.side_effect = combine
        with patch.object(token_switch, "_import_nccl_ep", return_value=native):
            received, _, _ = token_switch._dispatch_async_op(
                routing._opaque_handle, tokens, weights, 2, "flat"
            )
            received.square().sum().backward()
        self.assertEqual(tokens.grad, 2 * tokens.detach())
        native._nccl_ep_combine.assert_called_once()
        native._nccl_ep_combine_backward.assert_not_called()
        self.assertIsNone(weights.grad)

    @parametrize("async_path", [False, True])
    def test_combine_backward_preserves_padded_rows(self, async_path):
        routing = self._routing()
        experts = torch.randn(5, 4, requires_grad=True)
        native = Mock()

        def combine(handle, expert_tokens, output, *, async_op):
            output.copy_(expert_tokens[:2])

        def dispatch_backward(handle, grad_tokens, output, *, async_op):
            output[:2].copy_(grad_tokens)

        native._nccl_ep_combine.side_effect = combine
        native._nccl_ep_dispatch_backward.side_effect = dispatch_backward
        with patch.object(token_switch, "_import_nccl_ep", return_value=native):
            if async_path:
                combined = token_switch._combine_async_op(
                    routing._opaque_handle, experts, 2, 2
                )
            else:
                switch = Mock(spec=token_switch.TokenSwitch)
                switch.max_recv_tokens_per_rank = 2
                switch._combine.side_effect = (
                    lambda routing, expert_tokens, output: output.copy_(
                        expert_tokens[:2]
                    )
                )
                switch._combine_backward.side_effect = (
                    lambda routing, grad_tokens, output: output[:2].copy_(grad_tokens)
                )
                combined = token_switch._CombineAutograd.apply(switch, routing, experts)
            combined.sum().backward()
        expected = torch.zeros_like(experts)
        expected[:2] = 1
        self.assertEqual(experts.grad, expected)
        if async_path:
            self.assertEqual(
                native._nccl_ep_dispatch_backward.call_args.args[2].shape,
                experts.shape,
            )
        else:
            self.assertEqual(
                switch._combine_backward.call_args.args[2].shape, experts.shape
            )

    @parametrize("received_weights", [False, True])
    def test_base_requested_router_gradients(self, received_weights):
        routing = self._routing()
        tokens = torch.randn(2, 4, requires_grad=True)
        weights = torch.randn(2, 1, requires_grad=True)
        switch = Mock(spec=token_switch.TokenSwitch)
        switch._alloc_dispatch_outputs.return_value = (
            torch.zeros_like(tokens),
            torch.zeros_like(weights) if received_weights else None,
            None,
        )

        def dispatch(routing, tokens, weights, output, scores, ids):
            output.copy_(tokens)
            if scores is not None:
                scores.copy_(weights)

        def backward(routing, grad_tokens, grad_scores, out_tokens, out_scores):
            out_tokens.copy_(grad_tokens)
            out_scores.copy_(grad_scores)

        def combine(routing, expert_tokens, output):
            output.copy_(expert_tokens)

        switch._dispatch.side_effect = dispatch
        switch._dispatch_backward.side_effect = backward
        switch._combine.side_effect = combine
        received, scores, _ = token_switch._DispatchAutograd.apply(
            switch, routing, tokens, weights, 2
        )
        if not received_weights:
            received.square().sum().backward()
            self.assertEqual(tokens.grad, 2 * tokens.detach())
            self.assertIsNone(weights.grad)
            switch._dispatch_backward.assert_not_called()
            switch._combine.assert_called_once()
        else:
            (received.square().sum() + scores.square().sum()).backward()
            self.assertEqual(tokens.grad, 2 * tokens.detach())
            self.assertEqual(weights.grad, 2 * weights.detach())
            switch._dispatch_backward.assert_called_once()
            switch._combine.assert_not_called()

    def test_backward_after_routing_update_raises(self):
        routing = token_switch.Routing(Mock(), torch.zeros(2, 1, dtype=torch.int64))
        tokens = torch.randn(2, 4, requires_grad=True)
        weights = torch.ones(2, 1)
        native = Mock()

        def dispatch(handle, tokens, weights, output, scores, ids, *, async_op):
            output.copy_(tokens)

        native._nccl_ep_dispatch.side_effect = dispatch
        with patch.object(token_switch, "_import_nccl_ep", return_value=native):
            received, _, _ = token_switch._dispatch_async_op(
                routing._opaque_handle, tokens, weights, 2, "flat"
            )
            updated = token_switch.TokenSwitchNCCL.update_routing(
                Mock(), routing, torch.ones(2, 1, dtype=torch.int64)
            )
            with self.assertRaisesRegex(RuntimeError, "routing was updated"):
                received.sum().backward()
            with self.assertRaisesRegex(RuntimeError, "routing was updated"):
                token_switch._combine_op(
                    routing._opaque_handle, received.detach(), torch.empty(2, 4)
                )
            token_switch._combine_op(
                updated._opaque_handle, received.detach(), torch.empty(2, 4)
            )
            again = token_switch.TokenSwitchNCCL.update_routing(
                Mock(), routing, torch.zeros(2, 1, dtype=torch.int64)
            )
        self.assertIs(again.handle, routing.handle)
        native._nccl_ep_combine.assert_called_once()
        native._nccl_ep_combine_backward.assert_not_called()


if __name__ == "__main__":
    run_tests()
