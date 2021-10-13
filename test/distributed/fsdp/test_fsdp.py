import functools
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._fsdp import FullyShardedDataParallel
from torch.distributed._fsdp.fully_sharded_data_parallel import TrainingState_
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# get full params of a model recursively
def _get_full_params(model, recurse=True):
    if recurse:
        # get all params for any nested FSDP instances.
        for module in model.modules():
            if isinstance(module, FullyShardedDataParallel):
                _get_full_params(module, recurse=False)
    else:
        torch.cuda.synchronize()
        model._rebuild_full_params()
        if model.module.flat_param is not None:
            model.module._unflatten_params()


class TransformerWithSharedParams(nn.Module):
    def __init__(
        self, group, *unused_args, d_vocab=23, d_model=16, add_bn=True, **unused_kwargs
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
    def __init__(self, group, wrap_fsdp, wrap_everything=False):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FullyShardedDataParallel(layer, group)
            return layer

        torch.manual_seed(0)  # keep everything deterministic

        if wrap_everything:
            self.module = nn.Sequential(
                _maybe_wrap(nn.Linear(8, 4)),
                _maybe_wrap(nn.Linear(4, 16)),
                _maybe_wrap(nn.Linear(16, 4)),
                _maybe_wrap(nn.Linear(4, 8)),
            )
        else:
            self.module = nn.Sequential(
                nn.Linear(8, 4),
                _maybe_wrap(
                    nn.Sequential(
                        _maybe_wrap(nn.Linear(4, 16)),
                        nn.Linear(16, 16),
                    )
                ),
                _maybe_wrap(nn.Linear(16, 4)),
                nn.Linear(4, 8),
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

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there is no enough GPUs.
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

    def _train_for_several_steps(self, model, num_steps, autocast, lr=0.01):
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
                loss = model.module.get_loss(input, output).to(model_device)
            assert (
                loss.dtype == torch.float32
            ), "loss data type should be float32, as the original \
                 parameter data type is float32."
            model.module.run_backward(loss)
            optim.step()
        if isinstance(model, FullyShardedDataParallel):
            model._assert_state(TrainingState_.IDLE)
        return loss.detach()

    def _test_identical_outputs(
        self, model_init_fn, num_steps=2, use_cuda=True, lr=0.01
    ):
        group = dist.distributed_c10d._get_default_group()
        rank = group.rank()
        # Establish reference behavior with PyTorch DDP (+ optionally autocast).
        model = model_init_fn(group=group, wrap_fsdp=False).cuda()
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )
        ref_loss = self._train_for_several_steps(
            model, num_steps, autocast=False, lr=lr
        )
        ref_full_params = list(model.parameters())

        # Confirm we get the same behavior using FullyShardedDataParallel.
        model = model_init_fn(group=group, wrap_fsdp=True)
        model = FullyShardedDataParallel(model)
        if use_cuda:
            model = model.cuda()
        else:
            assert next(model.parameters()).device == torch.device(
                "cpu"
            ), "module parameters should be placed on cpu if use_cuda is False."
        shard_loss = self._train_for_several_steps(
            model, num_steps, autocast=False, lr=lr
        )
        _get_full_params(model)
        shard_full_params = list(model.parameters())

        torch.testing.assert_allclose(ref_loss, shard_loss)
        self.assertEqual(
            ref_full_params,
            shard_full_params,
            exact_device=True,
            msg="FullyShardedDataParallel didn't match PyTorch DDP",
        )


class TestParityWithDDP(FSDPTest):
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. FullyShardedDataParallel.
    """

    @skip_if_lt_x_gpu(2)
    def test_nested_wrapped_model(self):
        self._test_identical_outputs(NestedWrappedModule)

    @skip_if_lt_x_gpu(2)
    def test_nested_all_wrapped_model(self):
        model_fn = functools.partial(NestedWrappedModule, wrap_everything=True)
        self._test_identical_outputs(model_fn)

    @skip_if_lt_x_gpu(2)
    def test_transformer_parameterized(self):
        self._test_identical_outputs(TransformerWithSharedParams)


if __name__ == "__main__":
    run_tests()
