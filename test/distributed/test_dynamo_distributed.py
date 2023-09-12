# Owner(s): ["module: dynamo"]
import copy
import functools
from io import StringIO
from typing import List
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch._C import FileCheck
import torch._dynamo
from torch._dynamo.backends.distributed import DDPOptimizer
import torch._dynamo.test_case
from contextlib import contextmanager
from torch import nn
from torch._dynamo import config
from torch._dynamo.utils import same
from torch._dynamo.testing import collect_results
from torch._inductor.utils import has_triton
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, lambda_auto_wrap_policy
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    DynamoDistributedMultiProcTestCase,
    import_transformers_or_skip,
    skip_if_lt_x_gpu,
    requires_nccl,
    _dynamo_dist_per_rank_init,
)
import torch._dynamo.logging
from torch._dynamo.comptime import comptime

def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

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
        return self.net(inputs)

def get_model(device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5):
    m = ToyModel(in_feat=in_feat, hidden_feat=hidden_feat, out_feat=out_feat).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs



class ToyInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(100, 100), nn.Linear(100, 100)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)

class ToyOuterModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layers = [ToyInnerModel().to(device) for _ in range(2)]
        self.layers = nn.Sequential(self.layers[0], nn.ReLU(), self.layers[1], nn.ReLU())

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


def apply_fsdp_with_checkpointing(model, wrap_policy, checkpoint_policy, use_activation_checkpointing=True):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )
    model = FSDP(
        copy.deepcopy(model),
        auto_wrap_policy=wrap_policy,
        use_orig_params=True
    )
    if use_activation_checkpointing:
        checkpoint_wrapper_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=checkpoint_wrapper_fn, check_fn=checkpoint_policy,
        )
    return model



def get_custom_model(device):
    class MyCustomLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(512, 512))

        def forward(self, x):
            tmp = torch.mm(x, self.weight.t())
            # test an edge case where torch.where.scalar was decomposed to aten.where.self(tensor, tensor, tensor)
            # and the tensors T(0.4) and T(0.5) were not wrapped in FakeTensors during DDPOptimizer compilation
            return tmp + torch.where(tmp < 0.5, 0.3, 0.6)

    class MyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            return self.linear(x)

    class MyModule(torch.nn.Module):
        def __init__(self):
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
        from transformers import BertConfig, AutoModelForMaskedLM
    except ImportError as e:
        raise unittest.SkipTest("Unable to import transformers") from e

    batch_size, max_length, config, device = 4, 512, BertConfig(), f"cuda:{rank}"
    model = AutoModelForMaskedLM.from_config(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
    inputs = {'input_ids': input_ids, 'labels': decoder_ids}
    model.train()
    return model, inputs

class CheckSplitsCompiler:
    def __init__(self):
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
    def __init__(self, module):
        super().__init__()
        self.module = module
        bucket_cap_mb = 25
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
    correct_results = collect_results(model, correct_outputs.logits, correct_loss, inputs_flat)
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
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager_multiprocess(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            m = DDP(m, device_ids=[self.rank])
            m = torch._dynamo.optimize("aot_eager")(m)
            outputs = m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor(self):

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model)
            run_hf_bert_ddp(self, model, inputs, "inductor")

    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @patch.object(config, "optimize_ddp", True)
    def test_hf_bert_ddp_aot_eager(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model)
            run_hf_bert_ddp(self, model, inputs, "aot_eager")

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
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear, )
                ),
                use_orig_params=True
            )
            fsdp_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

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
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear, )
                ),
                use_orig_params=True
            )
            fsdp_m = torch._dynamo.optimize("inductor")(fsdp_m)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))


    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_activation_checkpointing(self):
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_toy_model_for_activation_checkpointing(f"cuda:{self.rank}")
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
            self.assertTrue(find_first_node(cnt.graphs[0], tag_activation_checkpoint) is not None)



    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_fsdp(self):

        def apply_fsdp(model, wrap_policy):
            model = FSDP(
                copy.deepcopy(model),
                auto_wrap_policy=wrap_policy,
                use_orig_params=True
            )
            return model

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for (wrap_policy, test_instance) in (
                (
                    None,
                    "FSDP without recursive wrapping"
                ),
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
                correct_results = collect_results(eager_model, correct_outputs.logits, correct_loss, inputs_flat)
                opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
                self.assertTrue(same(correct_results, opt_results))



    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_fsdp_activation_checkpointing(self):
        from transformers.models.bert.modeling_bert import BertLayer
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for (wrap_policy, test_instance) in (
                (
                    functools.partial(
                        transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer, )
                    ),
                    "FSDP with recursive wrapping BertLayer instances"
                ),
            ):
                print(f"Running hf_bert_activation_checkpointing test for {test_instance}")
                model, inputs = get_hf_bert(self.rank)
                check_fn = lambda submodule: isinstance(submodule, BertLayer)  # noqa: E731
                reset_rng_state()
                eager_model = apply_fsdp_with_checkpointing(model, wrap_policy, check_fn)
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
                correct_results = collect_results(eager_model, correct_outputs.logits, correct_loss, inputs_flat)
                opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
                self.assertTrue(same(correct_results, opt_results))


@requires_nccl()
class TestSingleProc(DynamoDistributedSingleProcTestCase):
    """
    Test harness initializes dist process group.

    Test simple things here since they are simpler to debug.
    Use TestMultiProc for things that really need to run on multiple nodes
    """

    def get_model(self, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5):
        m = ToyModel(in_feat=in_feat, hidden_feat=hidden_feat, out_feat=out_feat).to(self.device)
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

        # ensure compatibilty with dynamo explain

        explain_out = torch._dynamo.explain(ddp_m)(inputs)
        break_reasons = explain_out.break_reasons
        self.assertEqual(len(break_reasons), 3)
        self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor(self):
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

    @patch.object(config, "optimize_ddp", True)
    def test_no_split(self):
        """
        Ensures the DDPOptimizer returns a correct, compiled module without
        introducing graph splits. (Based on model parmeters fitting in the bucket)
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
            id(ddp_m.module.get_parameter(p))
            for p in ddp_m.parameters_to_ignore
        ]

        check_splits_compiler = CheckSplitsCompiler()
        ddp_optimizer = DDPOptimizer(
            bucket_bytes_cap=ddp_m.bucket_bytes_cap,
            backend_compile_fn=check_splits_compiler.compile_fn
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
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(N, N)
                self.linear2 = torch.nn.Linear(N, N)

            def forward(self, x):
                a = self.linear1(x)
                a = self.linear2(a)
                return a

        class MockModule(torch.nn.Module):
            def __init__(self):
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
            "DDPOptimizer backend: Found a higher order op in the graph"
        ):
            torch.compile(mod, backend=cnt)(*args)


    def test_fsdp_orig_params_assert(self):
        # Test with basic FSDP wrapping (outer wrap around whole model)
        m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
        fsdp_m = FSDP(m, use_orig_params=False)
        fsdp_m = torch._dynamo.optimize()(fsdp_m)
        self.assertRaisesRegex(AssertionError, "Dynamo only supports FSDP with use_orig_params=True", fsdp_m, inputs)

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
            (False, "local")
        ):
            torch._dynamo.reset()
            torch._dynamo.config.skip_fsdp_guards = skip_guards

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
            m = ToyModel(in_feat=10, hidden_feat=5000, out_feat=5,).to(device)
            inputs = torch.rand(20, 10).to(device)
            m.apply(init_weights)
            correct_outputs = m(inputs)
            fsdp_m = FSDP(m, use_orig_params=True)
            opt_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
            outputs = opt_m(inputs)

            # far from an exhaustive check of all the expected guards, just check a couple of them.
            FileCheck() \
                .check("""local "L['self']" TYPE_MATCH""") \
                .check("""local "L['self']" ID_MATCH""") \
                .check(f"""{expected_guard_source} "L['self'].net" TYPE_MATCH""") \
                .check(f"""{expected_guard_source} "L['self'].net" ID_MATCH""") \
                .check(f"""{expected_guard_source} "L['self'].net[0]" TYPE_MATCH""") \
                .check(f"""{expected_guard_source} "L['self'].net[0]" ID_MATCH""") \
                .run(GUARDS_FILE.getvalue())
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
                self.register_buffer(
                    "_buf", torch.randn((3,), requires_grad=False, device="cuda")
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
                self.register_buffer(
                    "_buf", torch.randn((3,), requires_grad=False, device="cuda")
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self._buf

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._param = nn.Parameter(torch.randn((1,), device="cuda"))
                self._buf_module = BufModule()
                # Share the buffer, meaning same tensor but different source
                self.register_buffer("_buf", self._buf_module._buf)

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
