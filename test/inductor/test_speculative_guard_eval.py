# Owner(s): ["module: inductor"]

import gc
import weakref

import torch
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor.speculative_guard import _is_zero_dropout_attention
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class InductorSpeculativeGuardEvalTests(TestCase):
    @parametrize("dropout_p, expected", [(0.0, True), (0.5, False)])
    def test_attention_dropout_classification(self, device, dropout_p, expected):
        graph = torch.fx.Graph()
        query = graph.placeholder("query")
        key = graph.placeholder("key")
        value = graph.placeholder("value")
        node = graph.call_function(
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            (query, key, value, dropout_p, False, False),
        )
        self.assertEqual(_is_zero_dropout_attention(node), expected)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_pure_inference_graph_attaches_descriptor(self, device):
        def fn(x):
            return x.sin() + 1

        x = torch.randn(32, device=device)
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x), fn(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 1)
        self.assertIsNotNone(entries[0].speculation_descriptor)

        descriptor = entries[0].speculation_descriptor
        commits = 0
        original_commit = descriptor.commit

        def count_commit(ticket):
            nonlocal commits
            commits += 1
            original_commit(ticket)

        descriptor.commit = count_commit
        with torch.inference_mode():
            self.assertEqual(compiled(x), fn(x))
        self.assertEqual(commits, 1)

        with torch.inference_mode():
            self.assertIsNone(
                descriptor.launch({"x": torch.randn(16, device=device)})
            )
            sparse = torch.sparse_coo_tensor(
                torch.empty((1, 0), dtype=torch.int64, device=device),
                torch.empty(0, device=device),
                (32,),
                device=device,
            )
            self.assertIsNone(descriptor.launch({"x": sparse}))

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_mutated_input_does_not_attach_descriptor(self, device):
        def fn(x):
            x.add_(1)
            return x

        x = torch.randn(32, device=device)
        expected = x + 1
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x), expected)

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_training_graph_does_not_attach_descriptor(self, device):
        def fn(x):
            return x.sin() + 1

        x = torch.randn(32, device=device, requires_grad=True)
        compiled = torch.compile(fn, fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_cached_bytecode_reconstructs_structured_output(self, device):
        def fn(x):
            return {"result": (x.sin() + 1, x.cos() - 1)}

        x = torch.randn(32, device=device)
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x), fn(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNotNone(entries[0].speculation_descriptor)

        with torch.inference_mode():
            self.assertEqual(compiled(x), fn(x))

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_random_graph_does_not_attach_descriptor(self, device):
        def fn(x):
            return x + torch.rand_like(x)

        x = torch.randn(32, device=device)
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            compiled(x)

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_indirect_indexing_does_not_attach_descriptor(self, device):
        def fn(x, index):
            return torch.gather(x, 1, index)

        x = torch.randn(4, 8, device=device)
        index = torch.zeros(4, 2, dtype=torch.int64, device=device)
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x, index), fn(x, index))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_guard_miss_aborts_before_recompile(self, device):
        def fn(x, add):
            if add:
                return x + 1
            return x - 1

        x = torch.randn(32, device=device)
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x, True), x + 1)

        entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)[0]
        descriptor = entry.speculation_descriptor
        self.assertIsNotNone(descriptor)
        aborts = 0
        original_abort = descriptor.abort

        def count_abort(ticket):
            nonlocal aborts
            aborts += 1
            original_abort(ticket)

        descriptor.abort = count_abort
        with torch.inference_mode():
            self.assertEqual(compiled(x, False), x - 1)
        self.assertEqual(aborts, 1)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {
            "compile_threads": 1,
            "force_disable_caches": True,
            "triton.cudagraphs": True,
        }
    )
    def test_cudagraph_does_not_attach_descriptor(self, device):
        def fn(x):
            return x.sin() + 1

        x = torch.randn(32, device=device)
        compiled = torch.compile(fn, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x), fn(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)

    @torch._dynamo.config.patch(speculative_guard_eval=True)
    @torch._inductor.config.patch(
        {"compile_threads": 1, "force_disable_caches": True}
    )
    def test_replaced_parameter_does_not_reuse_stale_output(self, device):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(8, 8, device=device), requires_grad=False
                )

            def forward(self, x):
                return x @ self.weight

        model = Model()
        x = torch.randn(8, 8, device=device)
        compiled = torch.compile(model, fullgraph=True)
        with torch.inference_mode():
            self.assertEqual(compiled(x), model(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(Model.forward)
        self.assertIsNotNone(entries[0].speculation_descriptor)

        descriptor = entries[0].speculation_descriptor
        model.weight.data = torch.randn(
            4, 8, device=device, dtype=model.weight.dtype
        )
        self.assertIsNone(descriptor.launch({"self": model, "x": x}))
        model.weight.data = torch.randn(
            8, 8, device=device, dtype=model.weight.dtype
        )

        old_weight = model.weight
        old_weight_ref = weakref.ref(old_weight)
        model.weight = torch.nn.Parameter(
            torch.full((8, 8), 2, device=device, dtype=model.weight.dtype),
            requires_grad=False,
        )
        calls = 0
        original_compiled_fn = descriptor.compiled_fn

        def count_call(*args):
            nonlocal calls
            calls += 1
            return original_compiled_fn(*args)

        descriptor.compiled_fn = count_call
        with torch.inference_mode():
            self.assertEqual(compiled(x), model(x))
        self.assertEqual(calls, 2)

        del old_weight
        gc.collect()
        self.assertIsNone(old_weight_ref())

    @functorch_config.patch(enable_autograd_cache=True)
    @torch._inductor.config.patch(
        {
            "compile_threads": 1,
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_aot_cache_preserves_eligibility(self, device):
        def fn(x):
            return x.sin() + 1

        x = torch.randn(32, device=device)
        with fresh_cache(), torch.inference_mode():
            with torch._dynamo.config.patch(speculative_guard_eval=False):
                torch.compile(fn, fullgraph=True)(x)
                torch.cuda.synchronize()
                entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
                self.assertIsNone(entries[0].speculation_descriptor)

            torch._dynamo.reset()
            cache_hits = counters["aot_autograd"]["autograd_cache_hit"]
            with torch._dynamo.config.patch(speculative_guard_eval=True):
                compiled = torch.compile(fn, fullgraph=True)
                self.assertEqual(compiled(x), fn(x))
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"], cache_hits + 1
            )

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNotNone(entries[0].speculation_descriptor)

    @functorch_config.patch(enable_autograd_cache=True)
    @torch._inductor.config.patch(
        {
            "compile_threads": 1,
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_aot_cache_preserves_ineligibility(self, device):
        def fn(x, index):
            return torch.gather(x, 1, index)

        x = torch.randn(4, 8, device=device)
        index = torch.zeros(4, 2, dtype=torch.int64, device=device)
        with fresh_cache(), torch.inference_mode():
            with torch._dynamo.config.patch(speculative_guard_eval=False):
                torch.compile(fn, fullgraph=True)(x, index)
                torch.cuda.synchronize()

            torch._dynamo.reset()
            cache_hits = counters["aot_autograd"]["autograd_cache_hit"]
            with torch._dynamo.config.patch(speculative_guard_eval=True):
                compiled = torch.compile(fn, fullgraph=True)
                self.assertEqual(compiled(x, index), fn(x, index))
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"], cache_hits + 1
            )

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertIsNone(entries[0].speculation_descriptor)


instantiate_device_type_tests(
    InductorSpeculativeGuardEvalTests,
    globals(),
    only_for=("cuda",),
)


if __name__ == "__main__":
    run_tests()
