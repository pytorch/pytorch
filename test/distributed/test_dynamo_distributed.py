# Owner(s): ["module: dynamo"]
import copy
import functools
import logging
import os
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch._dynamo
from torch._dynamo.optimizations.distributed import DDPOptimizer
import torch._dynamo.test_case
import torch.distributed as dist
from contextlib import contextmanager
from torch import nn
from torch._dynamo import config
from torch._dynamo.utils import same
from torch._dynamo.testing import collect_results
from torch._inductor.utils import has_triton
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    import_transformers_or_skip,
    skip_if_lt_x_gpu,
    requires_nccl
)
import torch._dynamo.logging


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

def get_custom_model(device):
    class MyCustomLinear(torch.nn.Module):
        def __init__(self):
            super(MyCustomLinear, self).__init__()
            self.weight = nn.Parameter(torch.randn(512, 512))

        def forward(self, x):
            return torch.mm(x, self.weight.t())

    class MyLinear(torch.nn.Module):
        def __init__(self):
            super(MyLinear, self).__init__()
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            return self.linear(x)

    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            mods = [
                (MyLinear(), torch.nn.ReLU()),
                # sandwitch the custom in the middle so it comes before and after
                (MyCustomLinear(), torch.nn.ReLU()),
                (MyLinear(), torch.nn.ReLU()),
            ]
            self.seq = torch.nn.Sequential(*[x for items in mods for x in items])

        def forward(self, x):
            return self.seq(x)

    m = MyModule().to(device)
    m.apply(init_weights)
    inputs = torch.rand((512, 512)).to(device)
    correct_outputs = m(inputs)
    return m, inputs, correct_outputs

def get_hf_bert(rank):
    # Note: use @import_transformers_or_skip on your test case if you use this
    try:
        from transformers import BertConfig, AutoModelForMaskedLM
    except ImportError:
        unittest.skip("Unable to import transformers")

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

@contextmanager
def _per_rank_init(rank, world_size):
    # To avoid multiple inheritance from _dynamo.test_case.TestCase and MultiProcessTestCase,
    # Just manually implement the most important part of the dynamo behavior to reset/clear.
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6789'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    dist.destroy_process_group()


@requires_nccl()
class TestDistributedMultiProc(MultiProcessTestCase):
    def setUp(self):
        super(TestDistributedMultiProc, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        super(TestDistributedMultiProc, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        # Don't enable DDP + ReplicatedTensor, as that breaks Dynamo+DDP
        # TODO(whc) why is ReplicatedTensor defaulted=True in MultiProcessTestCase, and should we support it?
        # from torch.nn.parallel._replicated_tensor_ddp_utils import _set_ddp_with_replicated_tensor
        # _set_ddp_with_replicated_tensor(True)

        # The rest is copypasta from MultiProcessTestCase._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    @skip_if_lt_x_gpu(2)
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager_multiprocess(self):
        with _per_rank_init(self.rank, self.world_size):
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
    def test_hf_bert_ddp(self):

        with _per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model)

            reset_rng_state()
            correct_outputs = model(**inputs)
            correct_loss = correct_outputs.loss
            correct_loss.backward()

            reset_rng_state()
            opt_model = torch._dynamo.optimize("inductor")(model)
            opt_outputs = opt_model(**inputs)
            opt_loss = opt_outputs.loss
            opt_loss.backward()

            inputs_flat = [inputs[k] for k in inputs]
            correct_results = collect_results(model, correct_outputs.logits, correct_loss, inputs_flat)
            opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
            self.assertTrue(same(correct_results, opt_results))


    @skip_if_lt_x_gpu(1)
    # TODO(whc)  delete aot_eager test, if inductor test lands stably
    def test_fsdp_aot_eager(self):
        with _per_rank_init(self.rank, self.world_size):
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
        with _per_rank_init(self.rank, self.world_size):
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

    @import_transformers_or_skip()
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_fsdp(self):
        from transformers.models.bert.modeling_bert import BertLayer
        def apply_fsdp(model, wrap_policy):
            model = FSDP(
                copy.deepcopy(model),
                auto_wrap_policy=wrap_policy,
                use_orig_params=True
            )
            return model

        with _per_rank_init(self.rank, self.world_size):
            for (wrap_policy, test_instance) in (
                (
                    None,
                    "FSDP without recursive wrapping"
                ),
                (
                    functools.partial(
                        transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer, )
                    ),
                    "FSDP with recursive wrapping BertLayer instances"
                )
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

                opt_model = torch._dynamo.optimize("aot_eager")(opt_model)
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(eager_model, correct_outputs.logits, correct_loss, inputs_flat)
                opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
                self.assertTrue(same(correct_results, opt_results))


@requires_nccl()
class TestDistributed(torch._dynamo.test_case.TestCase):
    """
    Test harness initializes dist process group
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # _exit_stack is set up in TestCase
        cls._exit_stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12355",
                },
            )
        )
        cls._exit_stack.enter_context(patch.object(config, "log_level", logging.DEBUG))
        cls.rank = 0
        cls.device = f"cuda:{cls.rank}"
        cls.device_ids = None if "cuda" in cls.device else [cls.rank]
        dist.init_process_group("nccl", rank=cls.rank, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        super().tearDownClass()

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
            return ddp_m(inputs)

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
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 2)
        for b in ddp_optimizer.buckets:
            for p_id in b.param_ids:
                self.assertFalse(p_id in parameter_ids_to_ignore)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
