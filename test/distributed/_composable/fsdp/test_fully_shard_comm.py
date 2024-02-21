# Owner(s): ["oncall: distributed"]

import functools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed._composable import checkpoint
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp._fsdp_common import TrainingState
from torch.distributed._composable.fsdp._fsdp_param import ShardedState
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DoubleLinear,
    FSDPTest,
    patch_all_gather,
    patch_post_backward,
    patch_reduce_scatter,
    patch_unshard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)

# For recording FSDP events like unshard or post-backward
EventType = Tuple[str, str, TrainingState]


class TestFullyShardCommunication(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_communication_count(self):
        """
        Tests that FSDP issues the expected number of all-gathers and
        reduce-scatters during forward and backward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_communication_count,
        )

    def _test_communication_count(
        self,
        reshard_after_forward: Union[bool, int],
    ):
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        num_blocks = 0
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
                num_blocks += 1
        fully_shard_fn(model)
        # We construct `num_blocks` plus 1 FSDP states/communication groups

        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor
        reduce_scatter_count = all_gather_count = 0

        def all_gather_with_count(*args, **kwargs):
            nonlocal all_gather_count
            all_gather_count += 1
            return orig_all_gather(*args, **kwargs)

        def reduce_scatter_with_count(*args, **kwargs):
            nonlocal reduce_scatter_count
            reduce_scatter_count += 1
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        with patch_all_gather(all_gather_with_count), patch_reduce_scatter(
            reduce_scatter_with_count
        ):
            loss = model(inp)
        self.assertEqual(all_gather_count, num_blocks + 1)
        self.assertEqual(reduce_scatter_count, 0)
        all_gather_count = reduce_scatter_count = 0
        with patch_all_gather(all_gather_with_count), patch_reduce_scatter(
            reduce_scatter_with_count
        ):
            loss.sum().backward()
        if reshard_after_forward is False:
            self.assertEqual(
                all_gather_count,
                0,
                f"Expects 0 but got {all_gather_count} for reshard_after_forward={reshard_after_forward}",
            )
        else:
            # The root always does not reshard after forward
            self.assertEqual(all_gather_count, num_blocks)
        self.assertEqual(reduce_scatter_count, num_blocks + 1)


class TestFullyShardBackwardPrefetch(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_backward_prefetch(self):
        # Activation checkpointing should not affect the expected FSDP events
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": [None, "utils", "composable"],
            },
            self._test_backward_prefetch_forward_backward,
        )
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": [None, "utils", "composable"],
            },
            self._test_backward_prefetch_multi_forward,
        )
        self._test_backward_prefetch_unused_in_backward(True)

    def _test_backward_prefetch_forward_backward(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: Optional[str]
    ):
        n_layers = 3
        model, optim, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )
        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        # Check the order for normal 1 forward, 1 backward, 1 optimizer step
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            for iter_idx in range(3):
                loss = model(inp)
                expected_events = [
                    ("unshard", "", TrainingState.FORWARD),  # root
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1", TrainingState.FORWARD),
                    ("unshard", "layers.2", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                expected_events = [
                    # Root does not reshard after forward so there is no
                    # unshard event for it in backward
                    ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                    # Explicit backward prefetching moves the unshards early
                    # by one module (note how swapping each unshard down one
                    # event would give the natural event order)
                    ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                    ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                    ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                    ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                    ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                    ("post_backward", "", TrainingState.POST_BACKWARD),
                ]
                if reshard_after_forward is False:
                    # No reshard after forward means no backward unshards
                    expected_events = [e for e in expected_events if e[0] != "unshard"]
                self.assertEqual(events, expected_events)
                events.clear()
                optim.step()
                optim.zero_grad(set_to_none=(iter_idx % 2 == 0))

    def _test_backward_prefetch_multi_forward(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: Optional[str]
    ):
        n_layers = 3
        model, optim, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )
        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        # Check the order for multiple forwards before 1 backward
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            loss1 = model(inp)
            loss2 = model(inp)
            expected_events = [
                ("unshard", "", TrainingState.FORWARD),  # root
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
                # Root does not reshard after forward so there is not another
                # unshard event for it
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
            ]
            if reshard_after_forward is False:
                # No reshard after forward means no second set of unshards
                expected_events = expected_events[:-3]
            self.assertEqual(events, expected_events)
            events.clear()
            (loss1 + loss2).sum().backward()
            expected_events = [
                # Same as the single forward/backward case except the root's
                # post-backward does not run until the end of backward in the
                # final callback (since the input not requiring gradient means
                # that we do not have a tensor on which to hook for
                # post-backward)
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
            ]
            if reshard_after_forward is False:
                # No reshard after forward means no backward unshards
                expected_events = [e for e in expected_events if e[0] != "unshard"]
                # However, the post-backward reshards, so the second set of
                # unshards will run as real ops
            expected_events += [
                # Repeat the same pattern except with the root's post-backward
                # at the end since the final callback runs
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

    def _test_backward_prefetch_unused_in_backward(
        self, reshard_after_forward: Union[bool, int]
    ):
        """
        Test a model with a linear module then a split into two linear modules,
        where we run backward through one path first before the other, meaning
        that (1) onlyh one linear of the two split is used per backward and (2)
        the initial shared linear is used in both backwards.
        """
        dim = 8
        model = nn.Sequential(nn.Linear(dim, dim), DoubleLinear(dim))
        fully_shard(model[0], reshard_after_forward=reshard_after_forward)
        fully_shard(model[1].lin1, reshard_after_forward=reshard_after_forward)
        fully_shard(model[1].lin2, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        inp = torch.randn((4, dim), device="cuda")
        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            loss1, loss2 = model(inp)
            expected_events = [
                # Root has no parameters, so it does not have an unshard
                ("unshard", "0", TrainingState.FORWARD),
                ("unshard", "1.lin1", TrainingState.FORWARD),
                ("unshard", "1.lin2", TrainingState.FORWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

            model.set_is_last_backward(False)
            loss2.sum().backward(retain_graph=True)
            expected_events = [
                ("unshard", "1.lin2", TrainingState.PRE_BACKWARD),
                # Check that `1.lin1` is not prefetched since it is not used
                # for this backward
                ("post_backward", "1.lin2", TrainingState.POST_BACKWARD),
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

            model.set_is_last_backward(True)
            loss1.sum().backward()
            expected_events = [
                # Check that `1.lin2` is not unsharded
                ("unshard", "1.lin1", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin1", TrainingState.POST_BACKWARD),
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

    def _init_transformer(
        self,
        n_layers: int,
        reshard_after_forward: Union[bool, int],
        checkpoint_impl: Optional[str],
    ):
        model_args = ModelArgs(
            n_layers=n_layers, checkpoint_activations=(checkpoint_impl == "utils")
        )
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if checkpoint_impl == "composable":
                    checkpoint(module)
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        inp = torch.randint(
            0, model_args.vocab_size, (2, model_args.max_seq_len), device="cuda"
        )
        return model, optim, inp

    def _get_unshard_with_record(
        self, orig_unshard: Callable, events: List[EventType]
    ) -> Callable:
        def unshard_with_record(self, *args, **kwargs):
            nonlocal events
            if (
                self._all_gather_result is None
                and self._sharded_state != ShardedState.UNSHARDED
            ):  # skip no-ops
                events.append(("unshard", self._module_fqn, self._training_state))
            return orig_unshard(self, *args, **kwargs)

        return unshard_with_record

    def _get_post_backward_with_record(
        self, orig_post_backward: Callable, events: List[EventType]
    ) -> Callable:
        def post_backward_with_record(self, *args, **kwargs):
            nonlocal events
            ret = orig_post_backward(self, *args, **kwargs)
            # Use training state after running post-backward to check that the
            # state is transitioned to `POST_BACKWARD` as expected
            events.append(("post_backward", self._module_fqn, self._training_state))
            return ret

        return post_backward_with_record


if __name__ == "__main__":
    run_tests()
