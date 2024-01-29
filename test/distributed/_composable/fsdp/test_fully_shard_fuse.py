# Owner(s): ["oncall: distributed"]

from typing import List, Tuple, Union

import torch

from torch.distributed._composable.fsdp import FSDP, fully_shard
from torch.distributed._composable.fsdp._fsdp_common import TrainingState
from torch.distributed._composable.fsdp._fsdp_param import ShardedState
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    patch_post_backward,
    patch_unshard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFullyShardFusePrefetch(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fuse_backward_prefetch(self):
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_fuse_backward_prefetch,
        )

    def _test_fuse_backward_prefetch(self, reshard_after_forward: Union[bool, int]):
        n_layers = 3
        model_args = ModelArgs(n_layers=n_layers)
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        FSDP.fuse_comm_groups(model.layers[1], model.layers[2])
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")

        orig_unshard = FSDPParamGroup.unshard
        orig_post_backward = FSDPParamGroup._post_backward
        events: List[Tuple[str, str, TrainingState]] = []

        def unshard_with_record(self, *args, **kwargs):
            nonlocal events
            if (
                self._all_gather_result is None
                and self._sharded_state != ShardedState.UNSHARDED
            ):  # skip no-ops
                events.append(("unshard", self._module_fqn, self._training_state))
            return orig_unshard(self, *args, **kwargs)

        def post_backward_with_record(self, *args, **kwargs):
            nonlocal events
            ret = orig_post_backward(self, *args, **kwargs)
            # Use training state after running post-backward to check that the
            # state is transitioned to `POST_BACKWARD` as expected
            events.append(("post_backward", self._module_fqn, self._training_state))
            return ret

        # Check the order for normal 1 forward, 1 backward, 1 optimizer step
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            for iter_idx in range(3):
                loss = model(inp)
                expected_events = [
                    ("unshard", "", TrainingState.FORWARD),  # root
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1, layers.2", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                expected_events = [
                    # Root does not reshard after forward so there is no
                    # unshard event for it in backward
                    ("unshard", "layers.1, layers.2", TrainingState.PRE_BACKWARD),
                    # Explicit backward prefetching moves the unshards early
                    # by one module (note how swapping each unshard down one
                    # event would give the natural event order)
                    ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                    (
                        "post_backward",
                        "layers.1, layers.2",
                        TrainingState.POST_BACKWARD,
                    ),
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

        # Check the order for multiple forwards before 1 backward
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            loss1 = model(inp)
            loss2 = model(inp)
            expected_events = [
                ("unshard", "", TrainingState.FORWARD),  # root
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1, layers.2", TrainingState.FORWARD),
                # Root does not reshard after forward so there is not another
                # unshard event for it
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1, layers.2", TrainingState.FORWARD),
            ]
            if reshard_after_forward is False:
                # No reshard after forward means no second set of unshards
                expected_events = expected_events[:-2]
            self.assertEqual(events, expected_events)
            events.clear()
            (loss1 + loss2).sum().backward()
            expected_events = [
                # Same as before except the root's post-backward does not run
                # until the end of backward in the final callback (since the
                # input not requiring gradient means that we do not have a
                # tensor on which to hook for post-backward)
                ("unshard", "layers.1, layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.1, layers.2", TrainingState.POST_BACKWARD),
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
                ("unshard", "layers.1, layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("post_backward", "layers.1, layers.2", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()


if __name__ == "__main__":
    run_tests()
