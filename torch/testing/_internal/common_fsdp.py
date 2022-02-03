# Owner(s): ["oncall: distributed"]

from contextlib import suppress
from enum import Enum
import os
import sys
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    TrainingState_,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    get_cycles_per_ms,
)



class FSDPInitMode(Enum):
    # Move model to CUDA before wrap
    CUDA_BEFORE = 1
    # Move model to CUDA after wrap
    CUDA_AFTER = 2
    # Don't move model to CUDA at all.
    CUDA_NEVER = 3

# get full params of a model recursively. Note that if CPU offloading, it will
# also automatically move the parameters to GPU, due to _rebuild_full_params
# call.
def get_full_params(model, recurse=True):
    if recurse:
        # get all params for any nested FSDP instances.
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                get_full_params(module, recurse=False)
    else:
        torch.cuda.synchronize()
        model._rebuild_full_params()
        torch.cuda.synchronize()
        if model.module.flat_param is not None:
            model.module._unflatten_params()

def _maybe_cuda(model, move_to_cuda):
    return model.cuda() if move_to_cuda else model

def _maybe_wrap_fsdp(model, wrap_fsdp, *args, **kwargs):
    return (
        model if not wrap_fsdp
        else FullyShardedDataParallel(model, *args, **kwargs)
    )

class DummyProcessGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class TransformerWithSharedParams(nn.Module):
    def __init__(
        self, group, *args, d_vocab=23, d_model=16, add_bn=True,
        fsdp_init_mode=FSDPInitMode.CUDA_AFTER, **kwargs
    ):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        torch.manual_seed(0)  # keep everything deterministic
        assert (
            d_vocab >= 12
        ), "dim of vocab should be larger than 12, as we use torch.arange(12) as input"

        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=8,
            dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)

        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer(
            "vocab_bias", self.embed_tokens.weight.new_ones((d_model,))
        )
        self.register_buffer("long_buffer", torch.zeros_like(self.vocab_bias, dtype=torch.long))  # type: ignore[arg-type]

        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()
        move_to_cuda = fsdp_init_mode == FSDPInitMode.CUDA_BEFORE
        self = _maybe_cuda(self, move_to_cuda)

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.arange(12, device=device).view(6, self.bs)  # T x B
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)  # T x B
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)  # type: ignore[operator]
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum"
        )

    def run_backward(self, loss):
        loss.backward()


class NestedWrappedModule(nn.Module):
    def __init__(self, group, wrap_fsdp, *args, wrap_everything=False, fsdp_init_mode=FSDPInitMode.CUDA_AFTER, **kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = fsdp_init_mode == FSDPInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FullyShardedDataParallel(layer, group, *args, **kwargs)
            return layer

        torch.manual_seed(0)  # keep everything deterministic

        if wrap_everything:
            self.module = nn.Sequential(
                _maybe_wrap(_maybe_cuda(nn.Linear(8, 4), move_to_cuda)),
                _maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)),
                _maybe_wrap(_maybe_cuda(nn.Linear(16, 4), move_to_cuda)),
                _maybe_wrap(_maybe_cuda(nn.Linear(4, 8), move_to_cuda)),
            )
        else:
            self.module = nn.Sequential(
                _maybe_cuda(nn.Linear(8, 4), move_to_cuda),
                _maybe_wrap(
                    nn.Sequential(
                        _maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)),
                        _maybe_cuda(nn.Linear(16, 16), move_to_cuda),
                    ),
                ),
                _maybe_wrap(_maybe_cuda(nn.Linear(16, 4), move_to_cuda)),
                _maybe_cuda(nn.Linear(4, 8), move_to_cuda),
            )

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()


class ModuleWithDelay(nn.Module):
    def __init__(self, module, delay_after_loss_ms=0, delay_before_reduction_ms=0):
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms
        self.delay_before_reduction_ms = delay_before_reduction_ms
        self.module = module

    def get_input(self, device):
        return self.module.get_input(device)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed._reduce_scatter_base

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(
                    int(self.delay_before_reduction_ms * get_cycles_per_ms())
                )
            return orig_reduce_scatter(*args, **kwargs)

        with mock.patch(
            "torch.distributed._reduce_scatter_base", _delayed_reduce_scatter
        ):
            self.module.run_backward(loss)


class NestedWrappedModuleWithDelay(ModuleWithDelay):
    def __init__(
        self,
        group,
        wrap_fsdp,
        fsdp_init_mode=FSDPInitMode.CUDA_AFTER,
        cpu_offload=None,
        backward_prefetch=None,
        **kwargs
    ):
        super().__init__(
            NestedWrappedModule(
                group,
                wrap_fsdp,
                fsdp_init_mode=fsdp_init_mode,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch,
            ),
            **kwargs
        )


class DummyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MixtureOfExperts(NestedWrappedModule):
    def __init__(self, group, wrap_fsdp, *args, delay_before_free_ms=0, fsdp_init_mode=FSDPInitMode.CUDA_BEFORE, **kwargs):
        super().__init__(group, wrap_fsdp)
        self.group = group
        self.delay_before_free_ms = delay_before_free_ms
        self.wrap_fsdp = wrap_fsdp
        self.move_to_cuda = fsdp_init_mode == FSDPInitMode.CUDA_BEFORE
        # "expert" params are different on each rank
        torch.manual_seed(42 + group.rank())
        d_expert = 23
        d_shared = 12
        d_input = 8
        expert = _maybe_cuda(nn.Linear(d_expert, d_shared), self.move_to_cuda)

        self.num_expert_params = sum([p.numel() for p in expert.parameters()])
        for p in expert.parameters():
            p.expert = True  # type: ignore[attr-defined]

        # everything else is shared
        torch.manual_seed(0)

        shared = _maybe_cuda(nn.Linear(d_shared, d_expert), self.move_to_cuda)

        if wrap_fsdp:
            # we create a process group of size 1 for the expert params
            expert_group = torch.distributed.new_group(
                [group.rank()]
            )  # world size 1 means no shard
            expert = FullyShardedDataParallel(expert, expert_group, **kwargs)  # type: ignore[assignment]

            shared = FullyShardedDataParallel(shared, group, **kwargs)  # type: ignore[assignment]

        self.module = nn.Sequential(
            _maybe_cuda(nn.Linear(d_input, d_shared), self.move_to_cuda),
            shared,
            expert,
            _maybe_cuda(nn.Linear(d_shared, d_input), self.move_to_cuda)
        )

    def forward(self, x):
        if self.delay_before_free_ms > 0:
            expert = self.module[2]
            if isinstance(expert, FullyShardedDataParallel):
                orig_free_full_params = self.module[2]._free_full_params

                def _free_full_params_with_delay(*args):
                    torch.cuda._sleep(
                        int(self.delay_before_free_ms * get_cycles_per_ms())
                    )
                    return orig_free_full_params(*args)

                assert hasattr(
                    expert, "_free_full_params"
                ), "expert FSDP module should has _free_full_params attribute."
                with mock.patch.object(
                    expert, "_free_full_params", _free_full_params_with_delay
                ):
                    return self.module(x)

        return self.module(x)

    def run_backward(self, loss):
        loss.backward()

        # manually reduce gradients if not wrapped in FullyShardedDataParallel
        if not self.wrap_fsdp:
            with torch.no_grad():
                for p in self.parameters():
                    if hasattr(p, "expert"):
                        continue  # these params don't need grad reduction
                    p.grad.div_(self.world_size)
                    torch.distributed.all_reduce(p.grad, group=self.group)


class FSDPTest(MultiProcessTestCase):
    def setUp(self):
        super(FSDPTest, self).setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return torch.cuda.device_count() if torch.cuda.is_available() else 4

    @property
    def init_method(self):
        return "{}{file_name}".format(FILE_SCHEMA, file_name=self.file_name)

    def _check_cpu_offload(self, fsdp_model, cpu_offload):
        self.assertEqual(cpu_offload, fsdp_model.cpu_offload)

    def _check_backward_prefetch(self, fsdp_model, backward_prefetch):
        self.assertEqual(backward_prefetch, fsdp_model.backward_prefetch)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there is no enough GPUs.

        backend = os.environ.get("BACKEND", None)
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        if torch.cuda.is_available() and torch.cuda.device_count():
            torch.cuda.set_device(self.rank % torch.cuda.device_count())

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        self.run_test(test_name, pipe)

        dist.barrier()

        dist.destroy_process_group()
        sys.exit(0)

    def _train_for_several_steps(self, model, num_steps, autocast, lr=0.01, fsdp_cpu_offload=None):
        cpu_offload_params = fsdp_cpu_offload and fsdp_cpu_offload.offload_params

        model_device = next(model.parameters()).device
        # use SGD with momentum instead of Adam, since Adam is scale invariant
        # and this makes it bad for tests
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                # Inputs always cuda regardless of cpu offloading, or model.device
                input = model.module.get_input(torch.device("cuda"))
                output = model(*input)
                # Post-forward, if CPU offloading model param should be on CPU.
                if cpu_offload_params and isinstance(model, FullyShardedDataParallel):
                    for p in model.parameters():
                        # Params should always be on CPU, even if
                        # p._is_sharded=False
                        self.assertEqual(p.device, torch.device("cpu"))

                loss = model.module.get_loss(input, output).to(model_device)
            assert (
                loss.dtype == torch.float32
            ), "loss data type should be float32, as the original \
                 parameter data type is float32."
            model.module.run_backward(loss)
            # Post-backward, if CPU offloading model params should be on CPU.
            if cpu_offload_params and isinstance(model, FullyShardedDataParallel):
                for p in model.parameters():
                    # Params should always be on CPU, even if
                    # p._is_sharded=False
                    self.assertEqual(p.device, torch.device("cpu"))
            optim.step()
        if isinstance(model, FullyShardedDataParallel):
            model._assert_state(TrainingState_.IDLE)
        return loss.detach()

    def _test_identical_outputs(
        self,
        model_init_fn,
        *args,
        ref_ddp_fn=None,
        num_steps=2,
        fsdp_init_mode=FSDPInitMode.CUDA_AFTER,
        lr=0.01,
        cpu_offload=CPUOffload(),
        backward_prefetch=None,
        **kwargs
    ):
        group = dist.distributed_c10d._get_default_group()
        rank = group.rank()
        # Establish reference behavior with PyTorch DDP (+ optionally autocast).
        model = model_init_fn(group=group, wrap_fsdp=False).cuda()
        if ref_ddp_fn is None:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )
        else:
            model = ref_ddp_fn(model)
        ref_loss = self._train_for_several_steps(
            model, num_steps, autocast=False, lr=lr, fsdp_cpu_offload=cpu_offload
        )
        ref_full_params = list(model.parameters())

        # Confirm we get the same behavior using FullyShardedDataParallel.
        try:
            model = model_init_fn(
                group=group,
                wrap_fsdp=True,
                fsdp_init_mode=fsdp_init_mode,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch
            )
        except Exception as e:
            raise ValueError(f"model_Init_fn {model_init_fn} got error {str(e)}")

        cpu_offload = cpu_offload or CPUOffload()  # disabled if not specified.
        model = FullyShardedDataParallel(model, cpu_offload=cpu_offload, backward_prefetch=backward_prefetch)
        # Call model.cuda() after init FSDP if specified.
        if fsdp_init_mode == FSDPInitMode.CUDA_AFTER:
            model = model.cuda()

        # Note that we don't do this check for FSDPInitMode.CUDA_AFTER since we
        # expect FSDP code to raise error that we check below, in the case of
        # offload params.
        if fsdp_init_mode != FSDPInitMode.CUDA_AFTER and cpu_offload.offload_params:
            for p in model.parameters():
                # Should be on CPU regardless of if param is sharded.
                self.assertEqual(p.device, torch.device("cpu"), f"Mismatch, cpu offload is {cpu_offload}")

        only_check_err = fsdp_init_mode == FSDPInitMode.CUDA_AFTER and cpu_offload.offload_params
        ctx = (
            self.assertRaisesRegex(AssertionError, "Expected param to be on CPU")
            if only_check_err else suppress()
        )
        with ctx:
            shard_loss = self._train_for_several_steps(
                model, num_steps, autocast=False, lr=lr,
                fsdp_cpu_offload=cpu_offload,
            )
        # We only check for errors in the case we have the following setup:
        # model = FSDP(model, cpu_offload=True)
        # model = model.cuda()
        # so skip the rest of this logic.
        if only_check_err:
            return
        # If CPU offload, next call will change model params to GPU. Sanity
        # check that params are on CPU before.
        if cpu_offload.offload_params:
            device_set = {p.device for p in model.parameters()}
            self.assertEqual(
                {torch.device("cpu")},
                device_set,
                f"Got device set {device_set}"
            )
        get_full_params(model)
        shard_full_params = list(model.parameters())

        if cpu_offload.offload_params:
            shard_loss = shard_loss.cuda()
        torch.testing.assert_allclose(ref_loss, shard_loss)
        self.assertEqual(
            ref_full_params,
            shard_full_params,
            exact_device=True,
            msg="FullyShardedDataParallel didn't match PyTorch DDP",
        )

    def _get_wrapped_model(
        self, group, cuda_first=False, **model_kwargs
    ) -> FullyShardedDataParallel:
        if cuda_first:
            model = FullyShardedDataParallel(
                TransformerWithSharedParams(group, **model_kwargs).cuda(), group
            )
        else:
            model = FullyShardedDataParallel(
                TransformerWithSharedParams(group, **model_kwargs), group
            ).cuda()
        return model
