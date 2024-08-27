# Owner(s): ["module: dynamo"]
import contextlib
import copy
import functools
import random
import unittest
from contextlib import contextmanager
from datetime import timedelta
from io import StringIO
from typing import List
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch._C import FileCheck
from torch._dynamo import config
from torch._dynamo.backends.distributed import DDPOptimizer
from torch._dynamo.comptime import comptime
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import same
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.distributed._functional_collectives import _maybe_wrap_tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    DynamoDistributedSingleProcTestCase,
    import_transformers_or_skip,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import requires_cuda
from torch.utils._triton import has_triton


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ToyModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        super().__init__()
        self.ctx_manager = ctx_manager
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )

    def forward(self, inputs):
        if self.ctx_manager is not None:
            with self.ctx_manager():
                return self.net(inputs)
        else:
            return self.net(inputs)


def get_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    m = ToyModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs


class MutatingModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        super().__init__()
        self.ctx_manager = ctx_manager
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )
        self.state = 1

    def forward(self, inputs):
        self.state = 2
        return self.net(inputs) * self.state


def get_mutating_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    m = MutatingModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs


class ToyInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(100, 100), nn.Linear(100, 100)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


class ToyOuterModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layers = [ToyInnerModel().to(device) for _ in range(2)]
        self.layers = nn.Sequential(
            self.layers[0], nn.ReLU(), self.layers[1], nn.ReLU()
        )

    def forward(self, inputs):
        return self.layers(inputs)


def get_toy_model_for_activation_checkpointing(device):
    m = ToyOuterModel(device).to(device)
    m.apply(init_weights)
    inputs = torch.rand(100, 100).to(device)
    return m, inputs


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def apply_fsdp_with_checkpointing(
    model, wrap_policy, checkpoint_policy, use_activation_checkpointing=True
):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    model = FSDP(
        copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
    )
    if use_activation_checkpointing:
        checkpoint_wrapper_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=checkpoint_policy,
        )
    return model


def get_custom_model(device):
    class MyCustomLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(512, 512))

        def forward(self, x):
            tmp = torch.mm(x, self.weight.t())
            # test an edge case where torch.where.scalar was decomposed to aten.where.self(tensor, tensor, tensor)
            # and the tensors T(0.4) and T(0.5) were not wrapped in FakeTensors during DDPOptimizer compilation
            return tmp + torch.where(tmp < 0.5, 0.3, 0.6)

    class MyLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            return self.linear(x)

    class MyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mods = [
                (MyLinear(), torch.nn.ReLU()),
                # sandwich the custom in the middle so it comes before and after
                (MyCustomLinear(), torch.nn.ReLU()),
                (MyLinear(), torch.nn.ReLU()),
            ]
            self.seq = torch.nn.Sequential(*[x for items in mods for x in items])

        def forward(self, x, y):
            # test special case where the 0th bucket (layers close to graph input) is at capacity, which would
            # trigger a new bucket, but there are only trivial ops without parameters to put into the new bucket.
            # optimize this case by fusing that 'empty bucket' back together with the previous full one
            return self.seq(x + y)

    m = MyModule().to(device)
    m.apply(init_weights)
    inputs = torch.rand((512, 512)).to(device)
    # test duplicated inputs
    inputs = (inputs, inputs)
    correct_outputs = m(*inputs)
    return m, inputs, correct_outputs


def get_hf_bert(rank):
    # Note: use @import_transformers_or_skip on your test case if you use this
    # in a multiprocessing test
    try:
        from transformers import AutoModelForMaskedLM, BertConfig
    except ImportError as e:
        raise unittest.SkipTest("Unable to import transformers") from e

    batch_size, max_length, config, device = 4, 512, BertConfig(), f"cuda:{rank}"
    model = AutoModelForMaskedLM.from_config(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
        device
    )
    inputs = {"input_ids": input_ids, "labels": decoder_ids}
    model.train()
    return model, inputs


class CheckSplitsCompiler:
    def __init__(self) -> None:
        self.compiler_called = 0

    def compile_fn(self, gm, example_inputs):
        self.compiler_called += 1
        return gm


# This simulates DDP, but it doesn't actually do any process communication;
# it just has enough properties so that the dynamo distributed optimization is
# able to optimize.  Feel free to simulate more properties as necessary.  The
# other important thing is patching _active_ddp_module, which is what actually
# triggers DDP optimization
class FakeDDP(nn.Module):
    def __init__(self, module, bucket_cap_mb=25):
        super().__init__()
        self.module = module
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

    @contextmanager
    def _inside_ddp_forward(self):
        DDP._active_ddp_module = self
        try:
            yield
        finally:
            DDP._active_ddp_module = None

    def forward(self, *inputs, **kwargs):
        with self._inside_ddp_forward():
            return self.module.forward(*inputs, **kwargs)


def run_hf_bert_ddp(self, model, inputs, backend):
    reset_rng_state()
    correct_outputs = model(**inputs)
    correct_loss = correct_outputs.loss
    correct_loss.backward()

    reset_rng_state()
    opt_model = torch._dynamo.optimize(backend)(model)
    opt_outputs = opt_model(**inputs)
    opt_loss = opt_outputs.loss
    opt_loss.backward()

    inputs_flat = [inputs[k] for k in inputs]
    correct_results = collect_results(
        model, correct_outputs.logits, correct_loss, inputs_flat
    )
    opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
    self.assertTrue(same(correct_results, opt_results))


class TestFakeDistributedSingleProc(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor(self):
        model, inputs = get_hf_bert(0)
        model = FakeDDP(model)
        run_hf_bert_ddp(self, model, inputs, "inductor")

    @patch.object(config, "optimize_ddp", True)
    def test_hf_bert_ddp_aot_eager(self):
        model, inputs = get_hf_bert(0)
        model = FakeDDP(model)
        run_hf_bert_ddp(self, model, inputs, "aot_eager")

    @patch.object(config, "optimize_ddp", True)
    def test_issue90375(self):
        class Model(nn.Module):
            def forward(self):
                return torch.randn(3) * torch.randn(3)

        model = Model()
        model = FakeDDP(model)

        opt_model = torch._dynamo.optimize("aot_eager")(model)
        opt_model()

    @patch.object(config, "optimize_ddp", True)
    def test_symbol_splitting(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x):
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = x + y @ self.weight2
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_direct(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                u0, u1 = y.tolist()
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * u0
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_indirect(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                u0, u1 = y.tolist()
                a = torch.ones(u0)
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * a.sum()
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_torture_multi(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))
                self.weight3 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                # partition one (contains the u0 def)
                u0, u1 = y.tolist()
                x = torch.cat([x, x])
                y1 = x @ self.weight1
                # partition two (contains the variable)
                y2 = y1 @ self.weight2
                a = torch.ones(u0)
                # partition three
                z = (x + y2 @ self.weight3) * a.sum()
                return z

        model = Model()
        model = FakeDDP(model, bucket_cap_mb=1)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/130534"
    @config.patch(optimize_ddp=True, capture_dynamic_output_shape_ops=True)
    def test_unbacked_symbol_splitting_no_binding(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                nz = y.nonzero()
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * (nz + 1).sum()
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([0.0, 12.0, 0.0, 11.0]))

    @patch.object(config, "optimize_ddp", True)
    def test_call_method_forward(self):
        class Model(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                layers = []
                for l in range(2):
                    layer = nn.ModuleList(
                        [
                            nn.LayerNorm(96),
                            nn.MultiheadAttention(
                                embed_dim=96, num_heads=4, batch_first=True
                            ),
                        ]
                    )
                    layers.append(layer)
                self.layers = nn.ModuleList(layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [Batch, Freq, Time, Feature]
                B, F, T, H = x.shape
                for m in self.layers:
                    x = x.reshape(B * F, T, H)
                    x = m[0](x)
                    x, attn = m[1].forward(x, x, x)
                    x = x.reshape(B, F, T, H)
                return x

        model = Model()
        model = FakeDDP(model)
        opt_model = torch.compile(model)
        opt_model(torch.randn(2, 129, 100, 96))


# Are these tests failing?  Check and see if TestFakeDistributedSingleProc has a
# single process version; if it's just a problem in the Dynamo distributed
# optimizer, you should be able to repro it single process!
@requires_nccl()
class TestMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """

    @skip_if_lt_x_gpu(2)
    @config.patch(optimize_ddp=False, enable_compiler_collectives=True)
    def test_ddp_baseline_aot_eager_multiprocess(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            m = DDP(m, device_ids=[self.rank])
            m = torch._dynamo.optimize("aot_eager")(m)
            outputs = m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    def _test_hf_bert_ddp_inductor(self, static_graph):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model, static_graph=static_graph)
            run_hf_bert_ddp(self, model, inputs, "inductor")

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor(self):
        self._test_hf_bert_ddp_inductor(static_graph=False)

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor_static_graph(self):
        self._test_hf_bert_ddp_inductor(static_graph=True)

    def _test_hf_bert_aot_eager(self, static_graph):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model, static_graph=static_graph)
            run_hf_bert_ddp(self, model, inputs, "aot_eager")

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    def test_hf_bert_ddp_aot_eager(self):
        self._test_hf_bert_aot_eager(static_graph=False)

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    def test_hf_bert_ddp_aot_eager_static_graph(self):
        self._test_hf_bert_aot_eager(static_graph=True)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=False, enable_compiler_collectives=True)
    def test_ddp_activation_checkpointing(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 32)
                self.fc2 = torch.nn.Linear(32, 16)
                self.fc3 = torch.nn.Linear(16, 8)

            def forward(self, inp):
                return self.fc3(self.fc2(self.fc1(inp)))

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            model = MyModel().to(device="cuda")

            # Activation checkpointing for Linear layers.
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(  # noqa: E731
                submodule, torch.nn.Linear
            )
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )

            model = DDP(model)
            x = torch.randn(10, 64).cuda()
            correct_outputs = model(x)

            opt_model = torch.compile(model)
            outputs = opt_model(x)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_aot_eager(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

            # Test with recursive wrapping, nested FSDP around each Linear
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            fsdp_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_setattr(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_mutating_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            prof = torch._dynamo.utils.CompileProfiler()
            fsdp_m = torch.compile(fsdp_m, backend=prof, fullgraph=False)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))
            FileCheck().check("Torchdynamo Profiler Report").check(
                "Graph Breaks"
            ).check_not(
                "setattr(FSDPManagedNNModuleVariable(MutatingModel), state, ...)"
            ).check_not(
                "setattr(FSDPManagedNNModuleVariable(FullyShardedDataParallel), _is_root, ...)"
            ).run(
                prof.report()
            )

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_inductor(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch._dynamo.optimize("inductor")(fsdp_m)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

            # Test with recursive wrapping, nested FSDP around each Linear
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            fsdp_m = torch._dynamo.optimize("inductor")(fsdp_m)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_activation_checkpointing(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_toy_model_for_activation_checkpointing(
                f"cuda:{self.rank}"
            )
            is_inner = lambda module: isinstance(module, ToyInnerModel)  # noqa: E731
            wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_inner)
            model = apply_fsdp_with_checkpointing(model, wrap_policy, is_inner)
            correct_outputs = model(inputs)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
            opt_model = torch._dynamo.optimize(cnt)(model)
            outputs = opt_model(inputs)
            self.assertTrue(same(correct_outputs, outputs))
            # Each FSDP module is a separate graph
            self.assertEqual(cnt.frame_count, 2)
            self.assertTrue(
                find_first_node(cnt.graphs[0], tag_activation_checkpoint) is not None
            )

    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @config.patch(enable_compiler_collectives=True)
    @unittest.skipIf(
        PLATFORM_SUPPORTS_FLASH_ATTENTION or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Inaccurate results with fused SDPA kernels",
    )
    def test_hf_bert_fsdp(self):
        def apply_fsdp(model, wrap_policy):
            model = FSDP(
                copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
            )
            return model

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for wrap_policy, test_instance in (
                (None, "FSDP without recursive wrapping"),
            ):
                print(f"Running hf_bert test for {test_instance}")
                model, inputs = get_hf_bert(self.rank)
                reset_rng_state()
                eager_model = apply_fsdp(model, wrap_policy)
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                reset_rng_state()
                opt_model = apply_fsdp(model, wrap_policy)
                opt_model = torch._dynamo.optimize("inductor")(opt_model)
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                self.assertTrue(same(correct_results, opt_results))

    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @config.patch(guard_nn_modules=True, enable_compiler_collectives=True)
    def test_hf_bert_fsdp_activation_checkpointing(self):
        from transformers.models.bert.modeling_bert import BertLayer

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for wrap_policy, test_instance in (
                (
                    functools.partial(
                        transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer,)
                    ),
                    "FSDP with recursive wrapping BertLayer instances",
                ),
            ):
                print(
                    f"Running hf_bert_activation_checkpointing test for {test_instance}"
                )
                model, inputs = get_hf_bert(self.rank)
                check_fn = lambda submodule: isinstance(  # noqa: E731
                    submodule, BertLayer
                )
                reset_rng_state()
                eager_model = apply_fsdp_with_checkpointing(
                    model, wrap_policy, check_fn
                )
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                reset_rng_state()
                opt_model = apply_fsdp_with_checkpointing(model, wrap_policy, check_fn)
                opt_model = torch._dynamo.optimize("inductor")(opt_model)
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                self.assertTrue(same(correct_results, opt_results))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_tensor(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):

            class SimpleModel(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.linear = nn.Linear(input_size, output_size)

                def forward(self, x):
                    return self.linear(x)

            torch._dynamo.utils.clear_compilation_metrics()

            model = SimpleModel(10, 2).to(self.rank)
            model.forward = torch.compile(model.forward)
            ddp_model = DDP(model, device_ids=[self.rank])

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

            def B(s):
                return [torch.randn(s, 10), torch.randint(0, 2, (s,))]

            if self.rank == 0:
                dataloader = [B(5), B(8), B(6)]
            else:
                dataloader = [B(6), B(6), B(3)]

            for data, labels in dataloader:
                data, labels = data.to(self.rank), labels.to(self.rank)
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_scalar(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            # TODO: This should be possible to do inside the function, but
            device = f"cuda:{self.rank}"

            @torch.compile()
            def f(x, y):
                return x + torch.ones(y, device=device).sum()

            if self.rank == 0:
                dataloader = [3, 3, 7]
            else:
                dataloader = [3, 4, 9]

            for data in dataloader:
                f(torch.randn(5, device=self.rank), data)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_speculation_divergence(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            # TODO: This should be possible to do inside the function, but
            device = f"cuda:{self.rank}"

            @torch.compile()
            def f(x, y):
                zx = x.shape
                zy = y.shape
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [4, 4]
            else:
                dataloader = [3, 4]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_graph_break_empty_graph_still_collective(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            device = f"cuda:{self.rank}"

            @torch.compile()
            def f(x, y):
                z = y
                print("woof")
                zx = x.shape
                zy = y.shape
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [5, 5, 6]
            else:
                dataloader = [3, 4, 5]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "fx_graph_cache", False)
    @patch.object(torch._inductor.config, "fx_graph_remote_cache", False)
    def test_asymmetric_compilation(self):
        from torch._dynamo.comptime import comptime

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            device = f"cuda:{self.rank}"

            pg = dist.distributed_c10d._get_default_group()

            cnt = torch._dynamo.testing.CompileCounter()
            sleep_time = 5

            @torch._dynamo.optimize(cnt)
            def f(x):
                if self.rank == 0:
                    comptime.sleep(sleep_time)

                y = 2 * x
                return y.sum()

            backend = pg._get_backend(torch.device(device))
            backend._set_default_timeout(timedelta(seconds=sleep_time - 2))

            x = torch.ones(4, device=device)

            # NCCL startup is lazy
            w = pg.allreduce(x)
            w.wait()

            f(x)
            if self.rank != 0:
                # test fails with NCCL timeout without this line
                dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                    timedelta(seconds=sleep_time)
                )

            w = pg.allreduce(x)
            w.wait()
            torch.cuda.synchronize(device)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "fx_graph_cache", True)
    @patch.object(torch._inductor.config, "fx_graph_remote_cache", False)
    @patch.object(torch._inductor.config, "sleep_sec_TESTING_ONLY", 10)
    def test_asymmetric_compilation_with_fx_cache(self):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_inductor_cache

        with fresh_inductor_cache(), _dynamo_dist_per_rank_init(
            self.rank, self.world_size
        ):
            torch._dynamo.utils.clear_compilation_metrics()

            device = f"cuda:{self.rank}"

            pg = dist.distributed_c10d._get_default_group()

            @torch.compile
            def f(x):
                y = 2 * x
                return y.sum()

            backend = pg._get_backend(torch.device(device))
            backend._set_default_timeout(timedelta(seconds=5))
            counters.clear()

            x = torch.ones(4, device=device)

            f(x)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            w = pg.allreduce(x)
            w.wait()
            torch.cuda.synchronize(device)
            torch._dynamo.reset()

            if self.rank == 0:
                with fresh_inductor_cache():
                    f(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            else:
                f(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            w = pg.allreduce(x)
            w.wait()
            torch.cuda.synchronize(device)


@requires_nccl()
@requires_cuda
class TestSingleProc(DynamoDistributedSingleProcTestCase):
    """
    Test harness initializes dist process group.

    Test simple things here since they are simpler to debug.
    Use TestMultiProc for things that really need to run on multiple nodes
    """

    def get_model(
        self, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
    ):
        m = ToyModel(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            out_feat=out_feat,
            ctx_manager=ctx_manager,
        ).to(self.device)
        m.apply(init_weights)
        inputs = torch.rand(bsz, in_feat).to(self.device)
        outputs = m(inputs)
        return m, inputs, outputs

    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager(self):
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch._dynamo.optimize("aot_eager")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_inductor(self):
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch._dynamo.optimize("inductor")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split(self):
        assert config.optimize_ddp
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

        # ensure compatibility with dynamo explain

        explain_out = torch._dynamo.explain(ddp_m)(inputs)
        break_reasons = explain_out.break_reasons
        self.assertEqual(len(break_reasons), 3)
        self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split_ctx_manager(self):
        """
        Ensures that we get the right number of splits and that the respective
        context managers' effects are applied to the computation.
        """

        for get_compiler in [
            lambda: CheckSplitsCompiler(),
            lambda: None,
        ]:
            for ctx_manager, output_test in [
                (
                    lambda: torch.autocast(
                        torch.device(self.device).type, torch.float16
                    ),
                    lambda out: self.assertEqual(out.dtype, torch.float16),
                ),
                (torch.enable_grad, lambda out: self.assertTrue(out.requires_grad)),
                (torch.no_grad, lambda out: self.assertTrue(not out.requires_grad)),
            ]:
                m, inputs, correct_outputs = self.get_model(
                    out_feat=1000,
                    hidden_feat=1000,
                    in_feat=1000,
                    ctx_manager=ctx_manager,
                )
                # inp - 1000 * 1000 matrix of float32 (4 bytes) = 4MB
                # hidden - 1000 * 1000 matrix of float32 (4 bytes) = 4MB
                bucket_cap_mb = 3.5  # 4MB
                ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=bucket_cap_mb)

                compiler = get_compiler()

                @torch._dynamo.optimize(
                    compiler.compile_fn if compiler else "aot_eager"
                )
                def opt_fn(inputs):
                    return ddp_m(inputs)

                opt_outputs = opt_fn(inputs)
                self.assertTrue(same(correct_outputs, opt_outputs))
                if compiler:
                    self.assertEqual(compiler.compiler_called, 4)

                output_test(opt_outputs)

                # ensure compatibility with dynamo explain

                explain_out = torch._dynamo.explain(ddp_m)(inputs)
                break_reasons = explain_out.break_reasons
                self.assertEqual(len(break_reasons), 4)
                self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor(self):
        assert config.optimize_ddp
        """
        Same as above, but using inductor backend.
        We observed issues with inductor/fx interface in the past.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch._dynamo.optimize("inductor")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))

    @torch._inductor.config.patch(
        {"layout_optimization": True, "keep_output_stride": False}
    )
    @patch.object(config, "optimize_ddp", True)
    def _test_graph_split_inductor_layout_optimizations_impl(self, context):
        assert config.optimize_ddp
        channel_dim = 512
        # channel dim must be > 64 for inductor to do layout optimization and use NHWC

        class ToyModelConv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    *[
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                )

            def forward(self, inputs):
                return self.net(inputs)

        def get_model():
            m = ToyModelConv().to(self.device)
            m.apply(init_weights)
            inputs = torch.rand(2, channel_dim, channel_dim, 128).to(self.device)
            outputs = m(inputs)
            return m, inputs, outputs

        with context():
            m, inputs, correct_outputs = get_model()
            ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

            @torch._dynamo.optimize("inductor")
            def opt_fn(inputs):
                return ddp_m(inputs)

            opt_outputs = opt_fn(inputs)
            self.assertTrue(same(correct_outputs, opt_outputs))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_layout_optimizations_training(self):
        self._test_graph_split_inductor_layout_optimizations_impl(
            contextlib.nullcontext
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_layout_optimizations_inference(self):
        self._test_graph_split_inductor_layout_optimizations_impl(torch.no_grad)

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_transpose(self):
        assert config.optimize_ddp

        B = 100
        N = 30
        D = 50
        K = 70

        class Foo(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear0 = nn.Linear(N, K)
                self.linear1 = torch.nn.Linear(D * K, 2048)

            def forward(self, x):
                xt = x.transpose(2, 1)
                xt = self.linear0(xt).flatten(1)
                return self.linear1(xt)

        mod = Foo().to(self.device)

        compiled_mod = torch.compile(mod, backend="inductor")
        ddp_compiled_mod = DDP(compiled_mod, device_ids=self.device_ids)

        x = torch.randn((B, N, D), dtype=torch.float32, device=self.device)
        self.assertTrue(same(mod(x), ddp_compiled_mod(x)))

        x_1 = torch.randn((B * 2, N, D), dtype=torch.float32, device=self.device)
        self.assertTrue(same(mod(x_1), ddp_compiled_mod(x_1)))

        x_2 = torch.randn((B * 3, N, D), dtype=torch.float32, device=self.device)
        self.assertTrue(same(mod(x_2), ddp_compiled_mod(x_2)))

    @patch.object(config, "optimize_ddp", True)
    def test_no_split(self):
        """
        Ensures the DDPOptimizer returns a correct, compiled module without
        introducing graph splits. (Based on model parameters fitting in the bucket)
        """
        # DDP will always do a 'first bucket' with a really small size;  so only a tiny model will escape this
        m, inputs, correct_outputs = self.get_model(hidden_feat=5)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=250)
        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 1)

    @patch.object(config, "optimize_ddp", True)
    def test_aot_autograd(self):
        """
        Explicitly check AotAutograd family of compilers work,
        since they require example inputs propagated between graph splits.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch._dynamo.optimize("aot_eager")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        opt_outputs.sum().backward()
        self.assertTrue(same(correct_outputs, opt_outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_custom_layer(self):
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """
        m, inputs, correct_outputs = get_custom_model(self.device)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=1)

        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(*inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_empty_graph_inductor(self):
        def fn():
            get_world_size = torch.distributed.distributed_c10d.get_world_size()
            return (get_world_size,)

        opt_fn = torch._dynamo.optimize("inductor")(fn)
        res = None
        try:
            res = opt_fn()[0]
        except Exception:
            pass
        self.assertEqual(res, 1)

    @patch.object(config, "optimize_ddp", False)
    def test_ignored_parameters(self):
        """
        Verifies ddp graph-split logic ignores parameters marked to ignore on DDP module.
        Hooks up graph-split optimizer manually so it can peek at internal state.
        """
        m, inputs, correct_outputs = get_custom_model(self.device)
        parameters_to_ignore = ["seq.2.weight", "seq.4.linear.bias"]
        DDP._set_params_and_buffers_to_ignore_for_model(m, parameters_to_ignore)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)
        parameter_ids_to_ignore = [
            id(ddp_m.module.get_parameter(p)) for p in ddp_m.parameters_to_ignore
        ]

        check_splits_compiler = CheckSplitsCompiler()
        ddp_optimizer = DDPOptimizer(
            bucket_bytes_cap=ddp_m.bucket_bytes_cap,
            backend_compile_fn=check_splits_compiler.compile_fn,
        )

        @torch._dynamo.optimize(ddp_optimizer.compile_fn)
        def opt_fn(inputs):
            return ddp_m(*inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 2)
        for b in ddp_optimizer.buckets:
            for p_id in b.param_ids:
                self.assertFalse(p_id in parameter_ids_to_ignore)

    @patch.object(config, "optimize_ddp", True)
    def test_higher_order_op(self):
        from torch.utils.checkpoint import checkpoint

        N = 1000

        class InnerModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(N, N)
                self.linear2 = torch.nn.Linear(N, N)

            def forward(self, x):
                a = self.linear1(x)
                a = self.linear2(a)
                return a

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_mod1 = InnerModule()
                self.inner_mod2 = InnerModule()

            def forward(self, x):
                a = checkpoint(self.inner_mod1, x, use_reentrant=False)
                a = torch.cos(a)
                a = checkpoint(self.inner_mod2, a, use_reentrant=False)
                a = torch.cos(a)
                return a

        mod = MockModule().cuda()
        mod = DDP(mod, bucket_cap_mb=1)
        x = torch.randn(N, N, device="cuda", requires_grad=True)
        args = (x,)

        backend = "aot_eager"
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "DDPOptimizer backend: Found a higher order op in the graph",
        ):
            torch.compile(mod, backend=cnt)(*args)

    def test_fsdp_orig_params_assert(self):
        # Test with basic FSDP wrapping (outer wrap around whole model)
        m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
        fsdp_m = FSDP(m, use_orig_params=False)
        fsdp_m = torch._dynamo.optimize()(fsdp_m)
        self.assertRaisesRegex(
            AssertionError,
            "Dynamo only supports FSDP with use_orig_params=True",
            fsdp_m,
            inputs,
        )

    def test_fsdp_skip_guards(self):
        """
        It's currently difficult to test dynamo guards.  Most guards tests are indirect- modify something and
        observe that the guard in question failed. In this case, since the FSDP guards were already deemed
        useless and skipping them is expected to have no practical effect, it's pretty contrived to even try to
        make those guards fail.  Instead, we observe the 'guard source' printed by dynamo's comptime print_guards
        function.

        Note: comptime prints the guards before the time they get installed or not installed, so in both cases
        (skip or no skip) the same guards get printed.  The difference is that in the skip case, they show up
        with a special 'guard source' which will cuase them to not be installed.  So all we check for is the expected
        guard source 'local_fsdp_module'.
        """
        global GUARDS_FILE
        GUARDS_FILE = StringIO()

        for skip_guards, expected_guard_source in (
            (True, "local_fsdp_module"),
            (False, "local_unspecialized_nn_module"),
        ):
            torch._dynamo.reset()

            class ToyModel(nn.Module):
                def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
                    super().__init__()
                    self.net = nn.Sequential(
                        *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
                    )

                def forward(self, inputs):
                    out = self.net(inputs)

                    @comptime
                    def _(ctx):
                        ctx.print_guards(file=GUARDS_FILE)

                    return out

            device = f"cuda:{self.rank}"
            m = ToyModel(
                in_feat=10,
                hidden_feat=5000,
                out_feat=5,
            ).to(device)
            inputs = torch.rand(20, 10).to(device)
            m.apply(init_weights)
            correct_outputs = m(inputs)
            fsdp_m = FSDP(m, use_orig_params=True)

            with torch._dynamo.config.patch(skip_fsdp_guards=skip_guards):
                opt_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
                outputs = opt_m(inputs)

            # far from an exhaustive check of all the expected guards, just check a couple of them.
            FileCheck().check("""local "L['self']" TYPE_MATCH""").check(
                f"""{expected_guard_source} "L['self']._modules['net']" TYPE_MATCH"""
            ).check(
                f"""{expected_guard_source} "L['self']._modules['net']._modules['0']" TYPE_MATCH"""
            ).run(
                GUARDS_FILE.getvalue()
            )

            self.assertTrue(same(correct_outputs, outputs))

    def test_fsdp_skip_register_attr_or_module(self):
        """
        ensure FSDP module is not registered as attrbutes
        in the fx graph
        see `not source.guard_source().is_fsdp_module()`
        before calling `register_attr_or_module`
        in variables/builder.py
        """

        class ToyModel(nn.Module):
            def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
                super().__init__()
                self.net = nn.Sequential(
                    *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
                    + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                )

            def forward(self, inputs):
                out = self.net(inputs)
                return out

        torch._dynamo.reset()

        device = f"cuda:{self.rank}"
        m = ToyModel(
            in_feat=10,
            hidden_feat=5000,
            out_feat=5,
        ).to(device)
        inputs = torch.rand(20, 10).to(device)
        m.apply(init_weights)
        correct_outputs = m(inputs)
        fsdp_m = FSDP(m, use_orig_params=True)

        def debug_compiler(gm, _):
            for node in gm.graph.nodes:
                if node.op == "get_attr":
                    for name in [
                        "l__self___net_0_weight",
                        "l__self___net_0_bias",
                        "l__self___net_2_weight",
                        "l__self___net_2_bias",
                    ]:
                        self.assertFalse(
                            name in node.name,
                            f"FSDP module {name} should not be registered as attributes",
                        )
            return gm

        opt_m = torch._dynamo.optimize(backend=debug_compiler)(fsdp_m)
        outputs = opt_m(inputs)

        self.assertTrue(same(correct_outputs, outputs))

    def test_fsdp_dup_tensors_same_source(self):
        """
        Tests that FSDP-managed modules' parameters and buffers with the same
        source are de-duplicated, meaning that they are each only passed once
        as a graph input.
        """

        class DuplicateModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._param = torch.randn((3,), device="cuda")
                self._buf = torch.nn.Buffer(
                    torch.randn((3,), requires_grad=False, device="cuda")
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Use `_param` and `_buf` each twice in this compiled forward
                # to exercise if they are de-duplicated by TorchDynamo
                z = x + self._buf + self._buf
                z += self._param + self._param
                return z

        model = DuplicateModule()
        fsdp_model = FSDP(copy.deepcopy(model), use_orig_params=True)
        fsdp_model = torch._dynamo.optimize("aot_eager")(fsdp_model)
        inp = torch.randn((2, 3), device="cuda")
        local_out = model(inp)
        fsdp_out = fsdp_model(inp)
        self.assertEqual(local_out, fsdp_out)

    @patch.object(config, "guard_nn_modules", True)
    def test_fsdp_dup_tensors_diff_source(self):
        """
        Tests that FSDP-managed modules' parameters and buffers with different
        source do not result in incorrect AOTAutograd de-dup guards like
        ``a is b``, where ``a`` and ``b`` are certainly not the same. We check
        this by checking for per-invocation recompiles.
        """

        class BufModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._buf = nn.Buffer(
                    torch.randn((3,), requires_grad=False, device="cuda")
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self._buf

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._param = nn.Parameter(torch.randn((1,), device="cuda"))
                self._buf_module = BufModule()
                # Share the buffer, meaning same tensor but different source
                self._buf = self._buf_module._buf

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Use the same buffer tensor twice in the compiled forward,
                # including a data mutation to trigger de-dup logic
                self._buf.mul_(2)
                z = x + self._buf
                z = self._buf_module(z)
                z += self._param
                return z

        fsdp_model = FSDP(Model(), use_orig_params=True)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        fsdp_model = torch._dynamo.optimize(cnt)(fsdp_model)
        inp = torch.randn((2, 3), device="cuda")
        for _ in range(15):
            fsdp_model(inp)
        # Check for no recompiles (if there were incorrect de-dup guards, then
        # the frame count would be equal to the number of forward calls)
        self.assertEqual(cnt.frame_count, 1)

    def test_fsdp_staticmethod(self):
        """
        Tests that Dynamo compiles staticmethods for FSDP-managed modules
        correctly both when the staticmethod is invoked from the class and from
        the object itself.
        """

        class ModuleWithStaticMethod(nn.Module):
            def __init__(self, use_self: bool):
                super().__init__()
                self._use_self = use_self
                torch.manual_seed(42)  # force `_param` to be deterministic
                self._param = nn.Parameter(torch.randn((3,), device="cuda"))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self._use_self:
                    z = self._add(x, self._param)
                else:
                    z = ModuleWithStaticMethod._add(x, self._param)
                z *= 2
                return z

            @staticmethod
            def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        model = ModuleWithStaticMethod(False)
        x = torch.randn((2, 3), device="cuda")
        ref_out = model(x)
        test_outs: List[torch.Tensor] = []

        for use_self in (False, True):
            model = ModuleWithStaticMethod(use_self)
            fsdp_model = FSDP(model, use_orig_params=True)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            fsdp_model = torch._dynamo.optimize(cnt)(fsdp_model)
            test_outs.append(fsdp_model(x))
            # Check for no recompiles, which could happen if incorrectly
            # passing args to the staticmethod (e.g. doubly passing `self`)
            # 3 is expected here for 1 forward.
            # Graph 1 should be add and imul
            self.assertEqual(cnt.frame_count, 1)
        for test_out in test_outs:
            self.assertEqual(test_out, ref_out)

    def test_async_subclass_no_specialize(self):
        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x):
            return x + 1

        f(_maybe_wrap_tensor(torch.randn(10)))
        f(_maybe_wrap_tensor(torch.randn(12)))

        self.assertEqual(cnt.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
