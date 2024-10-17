# Owner(s): ["oncall: distributed"]

import copy
import functools
import itertools
import unittest
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import (
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed._composable.fsdp._fsdp_collectives import (
    _div_if_needed,
    _get_gradient_divide_factors,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from torch.distributed._composable.fsdp._fsdp_common import FSDPMeshInfo, TrainingState
from torch.distributed._composable.fsdp._fsdp_init import (
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
)
from torch.distributed._composable.fsdp._fsdp_param import ShardedState
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.experimental import implicit_replication
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    DoubleLinear,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_post_backward,
    patch_reshard,
    patch_unshard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


c10d_ops = torch.ops.c10d

# For recording FSDP events like unshard or post-backward
EventType = Tuple[str, str, TrainingState]


class TestFullyShardCollectiveOps(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 128

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")

    def _get_param_sizes(self) -> List[torch.Size]:
        # For world size 128, the fp32 all-gather and reduce-scatter testing
        # requires ~0.22 GB
        return [
            torch.Size([17, 257]),
            torch.Size([17]),
            torch.Size([64, 312]),
            torch.Size([64]),
            torch.Size([64, 64]),
            torch.Size([512, 64]),
            torch.Size([256]),
            torch.Size([64, 297]),
        ]

    def _init_params(self, param_sizes: List[torch.Size]) -> List[nn.Parameter]:
        torch.manual_seed(42)
        orig_params = [
            nn.Parameter(torch.randn(size, device=self.device)) for size in param_sizes
        ]
        # Since seed is per process, not per thread, we broadcast to ensure the
        # same original parameters across ranks
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        return orig_params

    def _init_fsdp_param_group(
        self, params: List[nn.Parameter], reshard_after_forward: Union[bool, int]
    ):
        module = nn.ParameterList([param.detach().clone() for param in params])
        mesh_info = FSDPMeshInfo(_init_default_fully_shard_mesh(), shard_mesh_dim=0)
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward, mesh_info
        )
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()),
            (module,),
            mesh_info,
            post_forward_mesh_info,
            self.device,
            None,  # shard_placement_fn
            MixedPrecisionPolicy(),
            OffloadPolicy(),
        )
        fsdp_param_group.lazy_init()
        return fsdp_param_group

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.cuda.current_stream()
        stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
        for async_op, streams, reshard_after_forward in itertools.product(
            (False, True),
            ((default_stream, default_stream), (stream1, stream2)),
            (True, 8),
        ):
            all_gather_copy_in_stream, all_gather_stream = streams
            # Save test time by only testing reshard after forward as an int
            # for non-async and non-default streams (like in pre-backward)
            if type(reshard_after_forward) is int and (
                async_op or all_gather_stream is default_stream
            ):
                continue
            self._test_all_gather(
                param_sizes,
                reshard_after_forward=reshard_after_forward,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
            )

    def _test_all_gather(
        self,
        param_sizes: List[torch.Size],
        reshard_after_forward: Union[bool, int],
        async_op: bool,
        all_gather_copy_in_stream: torch.cuda.Stream,
        all_gather_stream: torch.cuda.Stream,
    ):
        def all_gather(fsdp_param_group: FSDPParamGroup, group: dist.ProcessGroup):
            all_gather_result = foreach_all_gather(
                fsdp_param_group.fsdp_params,
                group,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
                device=self.device,
            )
            foreach_all_gather_copy_out(all_gather_result, fsdp_params, group)
            # Transition to unsharded state to register unsharded parameters
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.init_unsharded_param()
            fsdp_param_group._to_unsharded()

        def check_all_gathered_params(
            orig_params: List[nn.Parameter], module: nn.Module
        ):
            for orig_param, param in zip(orig_params, module.parameters()):
                self.assertIsInstance(param, torch.Tensor)
                self.assertIsInstance(param, nn.Parameter)
                self.assertEqual(param, orig_param.to(param.dtype))

        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(
            orig_params, reshard_after_forward
        )
        fsdp_params = fsdp_param_group.fsdp_params
        module = fsdp_param_group.modules[0]

        # Sanity check that the parameter sharding is as expected
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, DTensor))
            self.assertEqual(param.full_tensor(), orig_param)

        # Run the foreach all-gather (including copy-in and copy-out)
        all_gather(fsdp_param_group, fsdp_param_group.mesh_info.shard_process_group)

        # Check all-gather correctness
        check_all_gathered_params(orig_params, module)

        # For reshard after after forward as an int, further test emulating the
        # pre-backward all-gather
        if type(reshard_after_forward) is not int:
            return
        fsdp_param_group._to_sharded_post_forward()
        all_gather(
            fsdp_param_group,
            fsdp_param_group.post_forward_mesh_info.shard_process_group,
        )
        check_all_gathered_params(orig_params, module)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_reduce_scatter_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float32,
            )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_reduce_scatter_fp16(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float16,
            )

    def _test_reduce_scatter(
        self,
        param_sizes: List[torch.Size],
        reduce_scatter_stream: torch.cuda.Stream,
        reduce_scatter_dtype: torch.dtype,
    ):
        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params, True)
        fsdp_params = fsdp_param_group.fsdp_params
        fsdp_param_group.comm_ctx.lazy_init(self.device)

        # Run one unshard to initialize metadata
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        fsdp_param_group.reshard()

        # Run the foreach reduce-scatter (including copy-in and view-out)
        torch.manual_seed(42)
        unsharded_grads = [torch.ones_like(param) * self.rank for param in orig_params]
        group = fsdp_param_group.mesh_info.shard_process_group
        self.assertEqual(group.size(), self.world_size)
        all_reduce_stream = torch.cuda.Stream()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            post_reduce_event,
            _,
        ) = foreach_reduce(
            fsdp_params,
            unsharded_grads,
            group,
            reduce_scatter_stream,
            orig_dtype=orig_params[0].dtype,
            reduce_dtype=reduce_scatter_dtype,
            device=self.device,
            reduce_scatter_reduce_op=None,
            all_reduce_group=None,
            all_reduce_stream=all_reduce_stream,
            all_reduce_grads=True,
            partial_reduce_output=None,
        )
        torch.cuda.current_stream().wait_event(post_reduce_event)

        # Check reduce-scatter correctness
        predivide_factor, postdivide_factor = _get_gradient_divide_factors(
            group, None, reduce_scatter_dtype
        )
        reduced_grads = [grad.detach().clone() for grad in unsharded_grads]
        for grad in reduced_grads:
            _div_if_needed(grad, predivide_factor)
            dist.all_reduce(
                grad,
                group=group,
                op=dist.ReduceOp.AVG if predivide_factor is None else dist.ReduceOp.SUM,
            )
            _div_if_needed(grad, postdivide_factor)
        for fsdp_param, reduced_grad in zip(fsdp_params, reduced_grads):
            sharded_grad = fsdp_param.sharded_param.grad
            self.assertIsInstance(sharded_grad, DTensor)
            self.assertEqual(sharded_grad.full_tensor(), reduced_grad)


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

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        with CommDebugMode() as fwd_comm_mode:
            loss = model(inp)
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        self.assertEqual(len(fwd_comm_counts), 1)
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_blocks + 1)
        with CommDebugMode() as bwd_comm_mode:
            loss.sum().backward()
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        if reshard_after_forward is False:
            self.assertEqual(len(bwd_comm_counts), 1)
        else:
            # The root always does not reshard after forward
            self.assertEqual(len(bwd_comm_counts), 2)
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_blocks)
        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_blocks + 1
        )

    @skip_if_lt_x_gpu(2)
    def test_manual_reshard_with_reshard_after_forward_false(self):
        """
        Tests that we can manually call ``reshard`` on FSDP modules that were
        initialized with ``reshard_after_forward=False`` and still run unshard.
        """
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=False)
        model = fully_shard(model, reshard_after_forward=False)
        num_fsdp_modules = sum(
            isinstance(module, FSDPModule) for module in model.modules()
        )

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        with CommDebugMode() as fwd_comm_mode:
            loss = model(inp)
        fwd_comm_counts = fwd_comm_mode.get_comm_counts()
        self.assertEqual(len(fwd_comm_counts), 1)
        self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        with CommDebugMode() as bwd_comm_mode:
            loss.sum().backward()
        bwd_comm_counts = bwd_comm_mode.get_comm_counts()
        self.assertEqual(len(bwd_comm_counts), 2)
        self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_fsdp_modules)
        self.assertEqual(
            bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_fsdp_modules
        )

    @skip_if_lt_x_gpu(2)
    def test_set_reduce_scatter_divide_factor(self):
        self.run_subtests(
            {"divide_factor": [self.world_size * 2, self.world_size]},
            self._test_set_reduce_scatter_divide_factor,
        )

    def _test_set_reduce_scatter_divide_factor(self, divide_factor: float):
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0, weight_tying=False)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=False)
        model = fully_shard(model, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.set_reduce_scatter_divide_factor(divide_factor)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")

        for iter_idx in range(10):
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                param.grad.mul_(1.0 / divide_factor)
                dist.all_reduce(param.grad)
            loss = model(inp).sum()
            loss.backward()
            ref_optim.step()
            optim.step()
            ref_optim.zero_grad()
            optim.zero_grad()
            self.assertEqual(ref_loss, loss)
            check_sharded_parity(self, ref_model, model)


class TestFullyShardPrefetch(FSDPTest):
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
        that (1) only one linear of the two split is used per backward and (2)
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
                # NOTE: This `1.lin1` unshard is a mistargeted prefetch.
                ("unshard", "1.lin1", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin2", TrainingState.POST_BACKWARD),
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

            model.set_is_last_backward(True)
            loss1.sum().backward()
            expected_events = [
                # NOTE: `1.lin1` is already unsharded from the mistargeted
                # prefetch in the first backward.
                # Prefetch `0`
                ("unshard", "0", TrainingState.PRE_BACKWARD),
                ("post_backward", "1.lin1", TrainingState.POST_BACKWARD),
                ("post_backward", "0", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_set_modules_to_forward_prefetch(self):
        n_layers = 4
        reshard_after_forward = True
        checkpoint_impl = "utils"
        model, _, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )

        def set_forward_prefetch(model: Transformer, num_to_prefetch: int) -> None:
            # Use model-specific knowledge to configure forward prefetching:
            # each transformer block (layer) prefetches for the next few
            for i, layer in enumerate(model.layers):
                if i >= len(model.layers) - num_to_prefetch:
                    break
                layers_to_prefetch = [
                    model.layers[i + j] for j in range(1, num_to_prefetch + 1)
                ]
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        reshard_with_record = self._get_reshard_with_record(
            FSDPParamGroup.reshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        expected_backward_events = [
            # Default backward prefetching
            ("unshard", "layers.3", TrainingState.PRE_BACKWARD),
            ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
            ("reshard", "layers.3", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.3", TrainingState.POST_BACKWARD),
            ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
            ("reshard", "layers.2", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
            ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
            ("reshard", "layers.1", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
            ("reshard", "layers.0", TrainingState.POST_BACKWARD),
            ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
            ("reshard", "", TrainingState.POST_BACKWARD),
            ("post_backward", "", TrainingState.POST_BACKWARD),
        ]
        with patch_unshard(unshard_with_record), patch_reshard(
            reshard_with_record
        ), patch_post_backward(post_backward_with_record):
            set_forward_prefetch(model, num_to_prefetch=1)
            loss = model(inp)
            expected_forward_events = [
                ("unshard", "", TrainingState.FORWARD),
                # `layers.i` prefetches `layers.i+1`
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("reshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.3", TrainingState.FORWARD),
                ("reshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.3", TrainingState.FORWARD),
            ]
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            self.assertEqual(events, expected_backward_events)
            events.clear()

            set_forward_prefetch(model, num_to_prefetch=2)
            loss = model(inp)
            expected_forward_events = [
                ("unshard", "", TrainingState.FORWARD),
                # `layers.i` prefetches `layers.i+1` and `layers.i+2`
                ("unshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.1", TrainingState.FORWARD),
                ("unshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.0", TrainingState.FORWARD),
                ("unshard", "layers.3", TrainingState.FORWARD),
                ("reshard", "layers.1", TrainingState.FORWARD),
                ("reshard", "layers.2", TrainingState.FORWARD),
                ("reshard", "layers.3", TrainingState.FORWARD),
            ]
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            self.assertEqual(events, expected_backward_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_set_modules_to_backward_prefetch(self):
        n_layers = 4
        reshard_after_forward = True
        checkpoint_impl = "utils"
        model, _, inp = self._init_transformer(
            n_layers, reshard_after_forward, checkpoint_impl
        )

        def set_backward_prefetch(model: Transformer, num_to_prefetch: int) -> None:
            # Use model-specific knowledge to configure backward prefetching:
            # each transformer block (layer) prefetches for the previous few
            for i, layer in enumerate(model.layers):
                if i < num_to_prefetch:
                    continue
                layers_to_prefetch = [
                    model.layers[i - j] for j in range(1, num_to_prefetch + 1)
                ]
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        reshard_with_record = self._get_reshard_with_record(
            FSDPParamGroup.reshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        expected_forward_events = [
            # Default forward prefetching
            ("unshard", "", TrainingState.FORWARD),  # root
            ("unshard", "layers.0", TrainingState.FORWARD),
            ("reshard", "layers.0", TrainingState.FORWARD),
            ("unshard", "layers.1", TrainingState.FORWARD),
            ("reshard", "layers.1", TrainingState.FORWARD),
            ("unshard", "layers.2", TrainingState.FORWARD),
            ("reshard", "layers.2", TrainingState.FORWARD),
            ("unshard", "layers.3", TrainingState.FORWARD),
            ("reshard", "layers.3", TrainingState.FORWARD),
        ]
        with patch_unshard(unshard_with_record), patch_reshard(
            reshard_with_record
        ), patch_post_backward(post_backward_with_record):
            set_backward_prefetch(model, num_to_prefetch=1)
            loss = model(inp)
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            expected_backward_events = [
                # Root prefetches `layers.3` per default
                ("unshard", "layers.3", TrainingState.PRE_BACKWARD),
                # `layers.i` prefetches for `layers.i-1` (same as default)
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.3", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.3", TrainingState.POST_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.2", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("reshard", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("reshard", "", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_backward_events)
            events.clear()

            set_backward_prefetch(model, num_to_prefetch=2)
            loss = model(inp)
            self.assertEqual(events, expected_forward_events)
            events.clear()
            loss.sum().backward()
            expected_backward_events = [
                # Root prefetches `layers.3` per default
                ("unshard", "layers.3", TrainingState.PRE_BACKWARD),
                # `layers.i` prefetches for `layers.i-1` and `layers.i-2`
                ("unshard", "layers.2", TrainingState.PRE_BACKWARD),
                ("unshard", "layers.1", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.3", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.3", TrainingState.POST_BACKWARD),
                ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                ("reshard", "layers.2", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.2", TrainingState.POST_BACKWARD),
                ("reshard", "layers.1", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.1", TrainingState.POST_BACKWARD),
                ("reshard", "layers.0", TrainingState.POST_BACKWARD),
                ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                ("reshard", "", TrainingState.POST_BACKWARD),
                ("post_backward", "", TrainingState.POST_BACKWARD),
            ]
            self.assertEqual(events, expected_backward_events)
            events.clear()

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_multi_module_backward_prefetch(self):
        n_layers = 5
        model_args = ModelArgs(n_layers=n_layers, checkpoint_activations=True)
        model = Transformer(model_args)
        for i in range(n_layers):
            if i == 0:
                fully_shard(model.layers[i])
            elif i % 2 == 1:
                fully_shard([model.layers[i], model.layers[i + 1]])
        fully_shard([model.tok_embeddings, model.pos_embeddings])
        fully_shard([model.norm, model.output], reshard_after_forward=False)
        fully_shard(model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        inp = torch.randint(
            0, model_args.vocab_size, (2, model_args.max_seq_len), device="cuda"
        )
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            for iter_idx in range(3):
                loss = model(inp)
                expected_events = [
                    (
                        "unshard",
                        "tok_embeddings, pos_embeddings",
                        TrainingState.FORWARD,
                    ),
                    ("unshard", "layers.0", TrainingState.FORWARD),
                    ("unshard", "layers.1, layers.2", TrainingState.FORWARD),
                    ("unshard", "layers.3, layers.4", TrainingState.FORWARD),
                    ("unshard", "norm, output", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                expected_events = [
                    # (norm, output) does not reshard after forward, so there is
                    # no unshard to begin backward
                    ("unshard", "layers.3, layers.4", TrainingState.PRE_BACKWARD),
                    ("post_backward", "norm, output", TrainingState.POST_BACKWARD),
                    ("unshard", "layers.1, layers.2", TrainingState.PRE_BACKWARD),
                    (
                        "post_backward",
                        "layers.3, layers.4",
                        TrainingState.POST_BACKWARD,
                    ),
                    ("unshard", "layers.0", TrainingState.PRE_BACKWARD),
                    (
                        "post_backward",
                        "layers.1, layers.2",
                        TrainingState.POST_BACKWARD,
                    ),
                    (
                        "unshard",
                        "tok_embeddings, pos_embeddings",
                        TrainingState.PRE_BACKWARD,
                    ),
                    ("post_backward", "layers.0", TrainingState.POST_BACKWARD),
                    (
                        "post_backward",
                        "tok_embeddings, pos_embeddings",
                        TrainingState.POST_BACKWARD,
                    ),
                ]
                events.clear()
                optim.step()
                optim.zero_grad()

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_multi_module_unused_module(self):
        class ModuleWithUnusedLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.unused_lin = nn.Linear(1, 1)
                self.lin = nn.Linear(16, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return nn.functional.relu(self.lin(x))

        model = nn.Sequential(
            ModuleWithUnusedLinear(), ModuleWithUnusedLinear(), nn.Linear(16, 16)
        )
        fully_shard([model[0].unused_lin, model[0].lin], reshard_after_forward=True)
        fully_shard([model[1].unused_lin, model[1].lin], reshard_after_forward=True)
        fully_shard(model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        events: List[EventType] = []
        unshard_with_record = self._get_unshard_with_record(
            FSDPParamGroup.unshard, events
        )
        post_backward_with_record = self._get_post_backward_with_record(
            FSDPParamGroup.post_backward, events
        )
        inp = torch.randn((2, 16), device="cuda")
        with patch_unshard(unshard_with_record), patch_post_backward(
            post_backward_with_record
        ):
            for iter_idx in range(3):
                loss = model(inp)
                expected_events = [
                    ("unshard", "", TrainingState.FORWARD),
                    ("unshard", "0.unused_lin, 0.lin", TrainingState.FORWARD),
                    ("unshard", "1.unused_lin, 1.lin", TrainingState.FORWARD),
                ]
                self.assertEqual(events, expected_events)
                events.clear()
                loss.sum().backward()
                expected_events = [
                    # Since both `model[0]` and `model[1]` have unused modules
                    # that never ran forward, they do not reshard after forward
                    # despite setting it to `True`. Check that there are no
                    # unshards in backward.
                    (
                        "post_backward",
                        "1.unused_lin, 1.lin",
                        TrainingState.POST_BACKWARD,
                    ),
                    (
                        "post_backward",
                        "0.unused_lin, 0.lin",
                        TrainingState.POST_BACKWARD,
                    ),
                    ("post_backward", "", TrainingState.POST_BACKWARD),
                ]
                events.clear()
                optim.step()
                optim.zero_grad()

    @skip_if_lt_x_gpu(2)
    def test_backward_misprefetch(self):
        torch.manual_seed(42)
        model = MLP(dim=16, device="cuda")
        ref_model = copy.deepcopy(model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model.in_proj)
        fully_shard(model.out_proj)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Backward should run through `out_proj` -> `in_proj`, so if `in_proj`
        # prefetches for `out_proj`, then this is a misprefetch, as `out_proj`
        # should not be needed anymore for backward.
        model.in_proj.set_modules_to_backward_prefetch([model.out_proj])

        torch.manual_seed(self.rank + 1)
        inp = torch.randn((2, 16), device="cuda")
        for _ in range(3):
            ref_optim.zero_grad()
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            ref_optim.step()
            optim.zero_grad()
            loss = model(inp).sum()
            loss.backward()
            optim.step()
            self.assertEqual(ref_loss, loss)

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

    def _get_reshard_with_record(
        self, orig_reshard: Callable, events: List[EventType]
    ) -> Callable:
        def reshard_with_record(self, *args, **kwargs):
            nonlocal events
            if (
                self._training_state == TrainingState.FORWARD
                and not self._reshard_after_forward
            ):  # skip no-ops
                return
            events.append(("reshard", self._module_fqn, self._training_state))
            return orig_reshard(self, *args, **kwargs)

        return reshard_with_record

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


class TestFullyShardUnshardMultiProcess(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_unshard_async(self):
        class ReduceModule(nn.Module):
            def __init__(self, dim: int, mesh: DeviceMesh):
                super().__init__()
                self.mesh = mesh
                self.weight = nn.Parameter(torch.randn(dim, dim))

            def forward(self, x: torch.Tensor):
                y = F.relu(x @ self.weight)
                # NOTE: This all-reduce is not differentiable and is included
                # to exercise the overlap.
                work = dist.all_reduce(y, group=self.mesh.get_group(), async_op=True)
                return y, work

        class MLPs(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.mlp1 = MLP(dim)
                self.mlp2 = MLP(dim)
                self.mlp3 = MLP(dim)

            def forward(self, ys: List[torch.Tensor], works: List[dist.Work]):
                (y1, y2, y3), (work1, work2, work3) = ys, works
                work1.wait()
                z1 = self.mlp1(y1)
                work2.wait()
                z2 = self.mlp2(y2)
                work3.wait()
                z3 = self.mlp3(y3)
                return z1 + z2 + z3

        class ReduceModel(nn.Module):
            def __init__(self, dim: int, mesh: DeviceMesh):
                super().__init__()
                self.reduce_module1 = ReduceModule(dim, mesh)
                self.reduce_module2 = ReduceModule(dim, mesh)
                self.reduce_module3 = ReduceModule(dim, mesh)
                self.mlps = MLPs(dim)

            def forward(self, x: torch.Tensor):
                y1, work1 = self.reduce_module1(x)
                if isinstance(self.mlps.mlp1, FSDPModule):
                    self.mlps.mlp1.unshard(async_op=True)
                y2, work2 = self.reduce_module2(x)
                if isinstance(self.mlps.mlp2, FSDPModule):
                    self.mlps.mlp2.unshard(async_op=True)
                y3, work3 = self.reduce_module3(x)
                if isinstance(self.mlps.mlp3, FSDPModule):
                    self.mlps.mlp3.unshard(async_op=True)
                return self.mlps([y1, y2, y3], [work1, work2, work3])

        mesh = init_device_mesh("cuda", (self.world_size,))
        batch_size, dim = 2, 8
        torch.manual_seed(42)
        ref_model = replicate(ReduceModel(dim, mesh).cuda())
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        torch.manual_seed(42)
        model = ReduceModel(dim, mesh)
        fully_shard(model.mlps.mlp1, reshard_after_forward=False)
        fully_shard(model.mlps.mlp2, reshard_after_forward=False)
        fully_shard(model.mlps.mlp3, reshard_after_forward=False)
        fully_shard(model.mlps)
        replicate(model.cuda())
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((batch_size, dim), device="cuda")
        for _ in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                losses.append(_model(inp).sum())
                losses[-1].backward()
                with implicit_replication():
                    _optim.step()
                _optim.zero_grad()
            self.assertEqual(losses[0], losses[1])


class TestFullyShardUnshardMultiThread(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_unshard_no_param_group(self):
        # Check that we can call `unshard()` on a module with no parameter
        # group / no managed parameters without erroring
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for lin in model:
            fully_shard(lin)
        fully_shard(model)
        handle = model.unshard(async_op=True)
        handle.wait()

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_unshard_without_lazy_init(self):
        torch.manual_seed(42)
        model = MLP(4)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(model)
        model.unshard()  # no lazy init yet
        for ref_param, param in zip(ref_model.parameters(), model.parameters()):
            self.assertEqual(ref_param, param)


if __name__ == "__main__":
    run_tests()
