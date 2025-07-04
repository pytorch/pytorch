# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import functools
import gc
import importlib
import itertools
import sys
import unittest
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch._dynamo.config as dynamo_config
import torch.nn as nn
from torch._dynamo.backends.debugging import aot_eager_decomp_partition_with_mode
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl
from torch._inductor.cudagraph_utils import FunctionID
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.immutable_collections import immutable_dict
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    skipIfRocm,
    TEST_CUDA_GRAPH,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("functorch")
importlib.import_module("filelock")

from torch.testing._internal.inductor_utils import HAS_CUDA


aten = torch.ops.aten
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from io import StringIO


def get_compile_fn(backend):
    if backend == "cudagraphs":
        return functools.partial(torch.compile, backend="cudagraphs")
    else:
        return functools.partial(torch.compile, mode="reduce-overhead")


class capture_stderr(list):
    """
    Replace sys.stderr with a temporary StringIO
    """

    def __enter__(self):
        self.sys_stderr = sys.stderr
        self.stringio = StringIO()
        sys.stderr = self.stringio
        return self

    def __exit__(self, *args):
        self.append(str(self.stringio.getvalue()))
        del self.stringio
        sys.stderr = self.sys_stderr


def cdata(t):
    return t.untyped_storage()._cdata


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()


if HAS_CUDA:

    def get_all_cudagraph_segments():
        segments = torch.cuda.memory_snapshot()
        return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]

    def all_live_blocks():
        blocks_addrs = []
        for segment in get_all_cudagraph_segments():
            addr = segment["address"]
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    blocks_addrs.append(addr)
                addr += block["size"]

        return blocks_addrs

    def all_live_block_count():
        return len(all_live_blocks())

    class CudaGraphTreeTests(TestCase):
        def setUp(self):
            super().setUp()
            self.graph_stack = contextlib.ExitStack()
            self.graph_stack.enter_context(
                config.patch(
                    {
                        "triton.cudagraphs": True,
                        "triton.cudagraph_trees": True,
                        "triton.fast_path_cudagraph_asserts": True,  # too slow
                        "triton.slow_path_cudagraph_asserts": True,
                    }
                )
            )
            self.graph_stack.enter_context(
                dynamo_config.patch(automatic_dynamic_shapes=True)
            )
            self.device_idx = torch.rand([0], device="cuda").device.index
            warnings.filterwarnings("ignore")

        def tearDown(self):
            super().tearDown()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            self.graph_stack.close()

            self.assertIsNone(self.get_manager())
            self.assertEqual(all_live_block_count(), 0)
            self.assertEqual(len(get_all_cudagraph_segments()), 0)
            warnings.resetwarnings()

        def get_manager(self, device_index=None):
            return torch._inductor.cudagraph_trees.get_container(
                self.device_idx if not device_index else device_index
            ).tree_manager

        def get_roots(self):
            return self.get_manager().get_roots()

        def curr_node(self):
            return self.get_manager().current_node

        def get_root_children(self):
            return [root.num_descendants() for root in self.get_roots()]

        def cudagraphify_impl(
            self, *args, is_inference=True, is_backward=False, **kwargs
        ):
            return tree_cudagraphify_impl(
                *args,
                **kwargs,
                device_index=self.device_idx,
                is_inference=is_inference,
                is_backward=is_backward,
            )

        @staticmethod
        def run_twc(fn, *args, **kwargs):
            fn(*args, **kwargs)
            return fn(*args, **kwargs)

        def num_checkpoints(self):
            return self.get_manager().debug_checkpointing_counter

        def test_run_simple(self):
            def foo(x):
                return x * x * x

            foo_opt = torch.compile(foo)
            ones = torch.ones([4, 4], device="cuda")
            zeros = torch.zeros([5, 5], device="cuda")
            self.run_twc(foo_opt, ones)
            self.run_twc(foo_opt, zeros)
            self.assertEqual(self.get_root_children(), [0, 0])

        def check_rng(self):
            @torch.compile(mode="reduce-overhead")
            def foo():
                return torch.rand([20])

            torch.manual_seed(0)

            out = foo()
            out2 = foo()
            out3 = foo()

            torch.manual_seed(0)

            self.assertEqual(out, foo())
            self.assertEqual(out2, foo())
            self.assertEqual(out3, foo())

        @torch._inductor.config.patch("fallback_random", True)
        def test_rng_trees(self):
            self.check_rng()

        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        @torch._inductor.config.patch("fallback_random", True)
        def test_rng_non_trees(self):
            self.check_rng()

        def test_mutation_reinplaced(self):
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def forward(self, input, other, out):
                    input = torch.logical_xor(input=input, other=other, out=out)
                    return input

            x = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32).cuda()
            y = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32).cuda()
            z = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float16).cuda()

            model = Model().cuda()
            eag = model(x, y, z)
            with capture_stderr() as captured_output:
                opt = torch.compile(model.forward, mode="reduce-overhead")(x, y, z)

            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check("torch.logical_xor").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @requires_multigpu()
        @parametrize("backend", ("inductor", "cudagraphs"))
        def test_multiple_devices_msg(self, backend):
            def foo(x, y):
                return (x + 1, y + 2)

            foo = get_compile_fn(backend)(foo)
            with capture_stderr() as captured_output:
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            FileCheck().check(
                "skipping cudagraphs due to cpu device (arg1_1). Found from"
            ).check("y + 2").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

            with capture_stderr() as captured_output:
                foo(
                    torch.ones([10], device="cuda:0"), torch.ones([10], device="cuda:1")
                )

            FileCheck().check("skipping cudagraphs due to multiple devices").run(
                captured_output[0]
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 2)

        @torch._inductor.config.patch("triton.cudagraph_skip_dynamic_graphs", True)
        def test_skip_symbolic(self):
            @torch.compile(dynamic=True)
            def foo(x, y):
                return x + y

            with capture_stderr() as captured_output:
                foo(torch.rand([10], device="cuda"), torch.rand([10], device="cuda"))

            FileCheck().check(
                "skipping cudagraphs due to graph with symbolic shapes inputs"
            ).check("x + y").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_on_inp(self, backend):
            def foo(x):
                x.add_(2)
                return x

            foo = get_compile_fn(backend)(foo)

            def inp():
                return torch.ones([10], device="cuda")

            with capture_stderr() as captured_output:
                foo(inp())

            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check(".add_(2)").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

            # mutation on inp doesnt hit cudagraphs
            self.assertEqual(len(self.get_manager().roots), 0)

            # mutation on parameters/buffers hits cudagraphs
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.buf = torch.ones([10], device="cuda")

                def forward(self, x):
                    self.buf.add_(x)
                    return self.buf + x

            def foo(mod, x):
                return mod(x)

            foo = get_compile_fn(backend)(foo)
            mod = Mod()
            mod2 = Mod()

            for _ in range(3):
                self.assertEqual(foo(mod, inp()), mod2(inp()))
                self.assertEqual(mod.buf, mod2.buf)

            self.assertIsNotNone(self.get_manager())

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", False)
        def test_mutation_cudagraph_managed_tensors_config(self, backend):
            def foo(x):
                return x + 1

            def mut(x):
                x.add_(2)
                return x

            def non_mut(x):
                return x.add(2)

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            with capture_stderr() as captured_output:
                for i in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.rand([4], device="cuda")

                    tmp = foo(inp)
                    mut_out = mut(tmp)
                    self.assertEqual(mut_out, non_mut(foo(inp)))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(captured_output[0])

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensors(self, backend):
            def foo(x):
                return x + 1

            def mut(x):
                x.add_(2)
                return x

            def non_mut(x):
                return x.add(2)

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            with capture_stderr() as captured_output:
                for i in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.rand([4], device="cuda")

                    tmp = foo(inp)
                    mut_out = mut(tmp)
                    self.assertEqual(mut_out, non_mut(foo(inp)))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                0,
                exactly=True,
            ).run(captured_output[0])
            self.assertTrue("cudagraph_skips" not in counters["inductor"])

            torch.compiler.cudagraph_mark_step_begin()
            inp = torch.rand([4], device="cuda")
            tmp = foo(inp)
            mut_inp = tmp.clone()
            # in this case, what previously a mutated cudagraph managed tensor is no longer,
            # now its an input from eager we should fallback to inductor without cudagraphs
            with capture_stderr() as captured_output:
                mut(mut_inp)
            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check("x.add_(2)").run(captured_output[0])
            self.assertEqual(mut_inp, non_mut(foo(inp)))
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensor_warn(self, backend):
            def foo(x):
                return x.add_(1)

            def fee(y, z):
                return z.add(3)

            def inp():
                return torch.rand([4], device="cuda")

            foo = get_compile_fn(backend)(foo)
            fee = get_compile_fn(backend)(fee)

            with capture_stderr() as captured_output:
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    fee(inp(), foo(inp()))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensor_warn_only_once(self, backend):
            def foo(x):
                return x + 1

            def mut(x):
                x.add_(2)
                return x

            def inp():
                return torch.rand([4], device="cuda")

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            with capture_stderr() as captured_output:
                # Should warn for current_node=None
                mut(inp())

                for i in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    tmp = foo(inp())
                    mut(tmp)  # should not warn

                mut_inp = tmp.clone()
                mut(mut_inp)  # should not warn since mut has warned

            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        def test_index_put(self):
            def fn(x, y, z):
                x = torch.zeros_like(x)
                return x.index_put_([y], z, True)

            fn_c = torch.compile(mode="reduce-overhead")(fn)

            for i in range(3):

                def args():
                    x = torch.zeros((512, 512), dtype=torch.bool, device="cuda")
                    y = torch.arange(512, dtype=torch.int64, device="cuda")
                    z = torch.ones((512, 512), dtype=torch.bool, device="cuda")
                    return x, y, z

                if i == 0:
                    out, code = run_and_get_code(fn_c, *args())
                    FileCheck().check("aten.index_put_").check_same("True").run(code[0])
                else:
                    out = fn_c(*args())

                self.assertEqual(fn(*args()), out)

        def test_function_compiled_multiple_times(self):
            def foo(x):
                y = foo2(x)
                y2 = foo2(y)
                return y + y2

            def foo2(x):
                torch._dynamo.graph_break()
                return x * x * x

            foo_opt = torch.compile(foo)
            ones = torch.ones([4, 4], device="cuda")
            foo(ones)
            foo_opt(ones)
            foo_opt(ones)
            self.assertEqual(foo_opt(ones), foo(ones))
            # paths
            children = self.get_root_children()
            # one root with two children
            self.assertEqual(children, [2])

        def test_end_recording_early(self):
            def foo(x):
                y = x * x * x
                torch._dynamo.graph_break()
                z = x + y
                return z

            @torch.compile
            def foo2(x):
                return x + 4

            foo_opt = torch.compile(foo)

            for _ in range(3):
                out = foo_opt(torch.ones([4, 4], device="cuda"))
                del out

                # when I tried inducing separate recordings via graph break,
                # the frame kept interferring by keeping outputs alive
                # this isnt great by simulates the logic.
                from torch._dynamo.mutation_guard import GenerationTracker

                GenerationTracker.generation -= 1

                out = foo2(torch.ones([4, 4], device="cuda"))
                del out

            foo_opt(torch.ones([4, 4], device="cuda"))

            # Two separate traces - one has a child, one doesnt
            self.assertEqual(self.get_root_children(), [1, 0])

        def test_execution_into_recording(self):
            def foo(x):
                y = x + x

                if y.sum() > 0:
                    return y + 10
                else:
                    return y - 10

            foo_opt = torch.compile(foo)
            inp = torch.zeros([4, 4], dtype=torch.float, device="cuda")
            self.assertEqual(foo_opt(inp), foo(inp))
            self.assertEqual(foo_opt(inp), foo(inp))

            inp.add_(1)
            out_eager = foo(inp)
            out_warmup = foo_opt(inp)
            self.assertEqual(out_warmup, out_eager)
            # warmup should be have storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)

            out_live = foo_opt(inp)
            self.assertEqual(out_live, out_eager)

            # should be in recording mode, with storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)
            # warmup should have been freed
            del out_warmup
            # should be in recording mode, with storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)

            del out_live
            self.assertEqual(all_live_block_count(), 0)

            out = foo_opt(inp)
            self.assertEqual(foo(inp), out)

            # should be in execution mode
            self.assertEqual(all_live_block_count(), 0)

        def test_forward_with_skipped_cudagraphed_backward(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            for _ in range(3):
                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)

                with config.patch(always_complex_memory_overlap_TESTING_ONLY=True):
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # we should not have cudagraph'd the backwards
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_cache_hit_forward_miss_backward(self):
            # Test that we don't cache cudagraphs, skipping cudagraphs on backward on a cache miss

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            # Run forwards, fx graph should cache miss
            for _ in range(3):
                torch._dynamo.reset()
                counters.clear()
                FxGraphCache.clear()
                AOTAutogradCache.clear()

                with config.patch(always_complex_memory_overlap_TESTING_ONLY=True):
                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

                    # Reset dynamo and related caches except for FXGraphCache
                    torch._dynamo.reset()
                    # Forwards should be a cache hit now, we still skip cudagraphs
                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

                    # Run backward without complex memory overlap being set

                # Run the backward without complex memory overlap reason
                # cache should miss, but cudagraphs should not run
                # because forward skipped it
                back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                out.backward(back_inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)

            # Run it one more time, this time AOTAutogradCache will hit
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            torch._dynamo.reset()
            inp = torch.rand([20, 20], device="cuda", requires_grad=True)
            out = foo(inp)
            back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
            out.backward(back_inp)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

            # we should not have cudagraph'd anything
            assert self.get_manager() is None

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        @requires_multigpu()
        def test_cached_boxed_forward_device_index(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            # Run with device index 1 so that we can see
            # on a cache hit we stay on device index 1
            with torch.cuda._DeviceGuard(1):
                torch.cuda.set_device(1)

                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                # Compile the backward and save to cache
                back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                out.backward(back_inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

                # Reset dynamo and rerun a few times
                for i in range(3):
                    torch._dynamo.reset()

                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    # Should cache hit each time; boxed_forward_device_index should still be set properly to 1
                    self.assertEqual(
                        counters["aot_autograd"]["autograd_cache_hit"], i + 1
                    )
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # After everything, we should have cudagraphs on device 1
            self.assertTrue(self.get_manager(device_index=0) is None)
            self.assertFalse(self.get_manager(device_index=1) is None)

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_backward_gets_cached_cudagraphs(self):
            # We pass cpu tensors to foo and save that into the cache
            # On a subsequent run in a new process, cudagraphs should be
            # disabled properly on both forward and backwards runs.

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            torch._dynamo.reset()
            counters.clear()
            FxGraphCache.clear()
            AOTAutogradCache.clear()

            # Use cpu device to disable cudagraphs during compilation
            inp = torch.rand([20, 20], device="cpu", requires_grad=True)
            out = foo(inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)

            # Run again on new process
            torch._dynamo.reset()

            # Forward and backward should also disable cudagraphs without compilation
            inp = torch.rand([20, 20], device="cpu", requires_grad=True)
            out = foo(inp)
            # AOTAutogradCache will load the forward and the backward from cache immediately, so fx_graph_cache_hit will equal 2
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            torch._dynamo.reset()

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)

            # we should not have cudagraph'd anything
            assert self.get_manager() is None

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_cached_forward_backward(self):
            counters.clear()
            AOTAutogradCache.clear()
            FxGraphCache.clear()

            @torch.compile
            def foo(x):
                torch.manual_seed(0)
                y = x * 2
                return torch.sin(y) * torch.nn.functional.dropout(x, p=0.4)

            inp = torch.rand([4, 4], requires_grad=True, device="cuda")
            inp2 = inp.detach().clone().requires_grad_(True)
            out = foo(inp)

            out.sum().backward()

            self.assertEqual(self.get_root_children(), [1])

            # the three saved tensors should die in the backward
            # we kept alive the output
            self.assertEqual(self.curr_node().expected_dead_indices_before_graph, [])
            self.assertEqual(
                self.curr_node().expected_dead_indices_after_graph,
                [(0, 1), (0, 2)],
            )
            self.assertFalse(self.get_manager().new_graph_id().id == 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)

            # Reset dynamo and rerun. We should see a cache hit now
            torch._dynamo.reset()

            out2 = foo(inp2)
            out2.sum().backward()
            self.assertEqual(out, out2)
            self.assertEqual(inp.grad, inp2.grad)

            self.assertEqual(self.get_root_children(), [1])
            self.assertFalse(self.get_manager().new_graph_id().id == 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        def test_forward_backward_not_called(self, backend):
            def foo(x, y):
                x_out = x * x * x
                torch._dynamo.graph_break()
                y_out = y * y * y
                return x_out, y_out

            foo = get_compile_fn(backend)(foo)

            for _ in range(3):
                inps = [
                    torch.rand([20, 20], requires_grad=True, device="cuda")
                    for _ in range(2)
                ]
                x_out, y_out = foo(inps[0], inps[1])
                x_out.sum().backward()

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            # we should not have cudagraph'd the y backward
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 3)

        def _test_unaligned_static_input_impl(self, expected_clones):
            def fn(x, y):
                return (x + y,)

            def get_aligned_inputs():
                return [torch.rand([5, 5], device="cuda") for _ in range(2)]

            mod = make_fx(fn)(*get_aligned_inputs())

            mode = torch._subclasses.FakeTensorMode()

            with mode:
                inps = [torch.rand([6, 5], device="cuda")[1:] for _ in range(2)]

            compiled_f = compile_fx_inner(
                mod, inps, static_input_idxs=[0], cudagraphs=True
            )

            def get_unaligned_inputs():
                return [torch.rand([6, 5], device="cuda")[1:] for _ in range(2)]

            class CloneCounterMode(TorchDispatchMode):
                def __init__(self) -> None:
                    self.count = 0

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = {} if kwargs is None else kwargs
                    self.count += func is torch.ops.aten.clone.default
                    return func(*args, **kwargs)

            for _ in range(3):
                with CloneCounterMode() as m:
                    compiled_f(get_unaligned_inputs())
                    self.assertEqual(m.count, expected_clones)

                    compiled_f(get_aligned_inputs())
                    self.assertEqual(m.count, expected_clones)

        def test_unaligned_static_input_trees(self):
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        def test_unaligned_static_input_non_trees(self):
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_unaligned_static_input_no_cudagraphs(self):
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        def test_graph_partition_gc(self):
            def _test_dummy():
                def foo(x):
                    return x + 1

                foo = torch.compile(foo)
                for _ in range(3):
                    foo(torch.randn(2, 3, device="cuda"))

            _test_dummy()
            gc.collect()
            self.assertIsNone(self.get_manager())

        def test_sparsity(self):
            def foo(view_6, buf31):
                return aten._sparse_coo_tensor_with_dims_and_tensors(
                    1,
                    1,
                    [1000000, 64],
                    view_6,
                    buf31,
                    dtype=torch.float32,
                    layout=torch.sparse_coo,
                    device="cuda",
                    pin_memory=None,
                )

            foo_opt = torch.compile(foo)

            view_6 = torch.zeros([1, 102397], dtype=torch.int64, device="cuda")
            buf31 = torch.rand([102397, 64], device="cuda")

            for _ in range(3):
                self.assertEqual(foo_opt(view_6, buf31), foo(view_6, buf31))

        def test_accumulate_multiple_recordings(self):
            def foo(x):
                y = x + x + x
                torch._dynamo.graph_break()
                if y.sum() <= 0:
                    return y
                else:
                    return y * 10

            foo_opt = torch.compile(foo)

            # two separate compilations & recordings
            out1 = self.run_twc(foo_opt, torch.zeros([5], device="cuda"))

            # out1 gets manually freed
            out2 = self.run_twc(foo_opt, torch.zeros([6], device="cuda"))

            self.assertEqual(all_live_block_count(), 1)

            out3 = self.run_twc(foo_opt, torch.ones([5], device="cuda"))

            self.assertEqual(out3, foo(torch.ones([5], device="cuda")))

            self.assertEqual(all_live_block_count(), 1)
            del out1, out2
            self.assertEqual(all_live_block_count(), 1)

            del out3
            gc.collect()
            self.assertEqual(all_live_block_count(), 0)

        @torch._inductor.config.patch("freezing", True)
        def test_constant_output(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(
                        torch.tensor([float(i) for i in range(10)], device="cuda")
                    )

                def forward(self, inp):
                    return self.param, self.param[0:2], inp + 2

            inp = torch.tensor([2], device="cuda")
            m = Mod()
            with torch.no_grad():
                out_eager = m(inp)

                m_comp = torch.compile(m)
                for _ in range(3):
                    self.assertEqual(out_eager, m_comp(inp))

        def test_live_outputs_multiple_graphs(self):
            def foo(x):
                x = x + x + x
                y = x + 1
                torch._dynamo.graph_break()
                z = x * x
                if z.sum() > 0:
                    return y + 1
                else:
                    return y

            foo_opt = torch.compile(foo)

            self.run_twc(foo_opt, torch.zeros([5], device="cuda"))
            self.assertEqual(self.num_checkpoints(), 0)
            out = self.run_twc(foo_opt, torch.ones([5], device="cuda"))

            self.assertEqual(all_live_block_count(), 1)

            del out
            self.assertEqual(all_live_block_count(), 0)

            # we need to checkpoint from function to warmup y + 1,
            # and then again to record it
            self.assertEqual(self.num_checkpoints(), 2)

        def test_expanded_inputs(self):
            x = torch.rand(1, 512, device="cuda").expand(4, 512)

            def foo(x):
                return x + 4 + torch.ones([4, 512], device="cuda")

            foo_opt = torch.compile()(foo)

            for _ in range(3):
                self.assertEqual(foo_opt(x), foo(x))

            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_dies_between_checkpoint(self):
            def foo(args):
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            foo_cg(inp_list)
            foo_cg([inp])

            out1, out2 = foo_cg([inp])
            inp = [out1]

            del out1, out2

            def foo2(args):
                x = args[0]
                args.clear()
                return [x * x * x]

            self.assertEqual(self.num_checkpoints(), 0)
            foo2_cg = self.cudagraphify_impl(foo2, inp, ())

            x = foo2_cg(inp)[0]

            self.assertEqual(self.num_checkpoints(), 1)
            # out2 dies between the previous recording and the new one,
            # need to be manually deallocated after the checkpoint

            self.assertEqual(all_live_block_count(), 1)
            del x
            self.assertEqual(all_live_block_count(), 0)

        def test_aliased_storage_single_weakref(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                x = x * 20
                x_alias = x[0]
                y = x * 10
                y_alias = y[0]
                torch._dynamo.graph_break()
                ind = torch.tensor(4, device="cuda")
                x_alias2 = x[ind:]
                y_alias2 = y[ind:]
                return x, x_alias, x_alias2, y_alias, y_alias2

            for _ in range(4):
                outs = foo(torch.rand([20, 20], device="cuda"))

                ptr_to_ref = {
                    out.untyped_storage().data_ptr(): out.untyped_storage()._cdata
                    for out in outs
                }

                self.assertEqual(len(ptr_to_ref), 2)
                for out in outs:
                    self.assertEqual(
                        ptr_to_ref[out.untyped_storage().data_ptr()],
                        out.untyped_storage()._cdata,
                    )
                del outs
                del out

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def test_aliasing_static_ref(self):
            class Mod(torch.nn.Linear):
                def forward(self, x):
                    return self.weight.T @ x, self.weight.T, self.weight[0:4]

            m = Mod(10, 10).cuda()

            @torch.compile(mode="reduce-overhead")
            def foo(mod, x):
                return mod(x)

            @torch.compile(mode="reduce-overhead")
            def foo2(x):
                return x[2:]

            param_c = cdata(m.weight)
            for _ in range(3):
                x = torch.rand([10, 10], device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out1, alias_1, alias_2 = foo(m, x)
                self.assertEqual(len({param_c, cdata(alias_1), cdata(alias_2)}), 1)

                out2 = foo2(out1)
                out2.sum().backward()
                self.assertEqual(cdata(out1), cdata(out2))
                m.weight.grad = None
                m.bias.grad = None

            node = self.curr_node()
            first_node = next(node._path_from_root)
            self.assertFalse(first_node.unaliased_in_all_paths[0])
            self.assertTrue(first_node.cached_tensor_outputs[0] is None)

        @torch._inductor.config.patch("implicit_fallbacks", True)
        def test_multinomial(self):
            def sample_multinomial(probs, num_samples, replacement=True):
                return torch.multinomial(probs, num_samples, replacement=replacement)

            # Create and prepare probability tensor on GPU
            probs = torch.tensor([0.1, 0.2, 0.3, 0.4]).cuda()
            probs = probs / probs.sum()

            # Sample using the function
            num_skipped = counters["inductor"]["cudagraph_skips"]

            with torch._dynamo.utils.preserve_rng_state():
                samples = self.run_twc(
                    sample_multinomial, probs, num_samples=5, replacement=True
                )

            with torch._dynamo.utils.preserve_rng_state():
                samples_compiled = self.run_twc(
                    torch.compile(sample_multinomial),
                    probs,
                    num_samples=5,
                    replacement=True,
                )

            self.assertEqual(samples, samples_compiled)
            self.assertEqual(num_skipped, counters["inductor"]["cudagraph_skips"])

        @skipIfRocm
        def test_checkpointing_resets_persistent_refs(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x @ x

            def inp():
                return torch.rand([20, 20], device="cuda", requires_grad=False)

            for _ in range(3):
                foo(inp())

            self.assertEqual(self.num_checkpoints(), 0)

            out = foo(inp())
            out_id = id(out)
            del out
            self.assertEqual(id(foo(inp())), out_id)

            @torch.compile(mode="reduce-overhead")
            def foo2(x):
                return x[0], x @ x

            for i in range(2):
                out = foo(inp())

                from torch._dynamo.mutation_guard import GenerationTracker

                GenerationTracker.generation -= 1

                out_alias, out2 = foo2(out)
                del out_alias

                self.assertEqual(all_live_block_count(), 2)
                del out
                self.assertEqual(all_live_block_count(), 1)
                del out2
                self.assertEqual(all_live_block_count(), 0)

                self.assertEqual(self.num_checkpoints(), i + 1)

            new_out = foo(inp())
            curr_node = self.curr_node()
            self.assertFalse(curr_node.unaliased_in_all_paths[0])
            self.assertFalse(out_id == id(new_out))

        def test_aliased_static_parameter(self):
            inp = torch.rand([20, 20], device="cuda")

            def foo(args):
                x = args[0]
                args.clear()
                return (x[0],)

            foo_cg = self.cudagraphify_impl(foo, [inp], (0,))

            for _ in range(3):
                out = foo_cg([inp])[0]
                self.assertEqual(cdata(inp), cdata(out))

            node = self.curr_node()
            self.assertEqual(node.cached_tensor_outputs, [None])
            self.assertEqual(node.unaliased_in_all_paths, [False])

        def test_warmup_stream_sync(self):
            def foo(args):
                x = args[0]
                args.clear()
                x_orig = x
                for _ in range(100):
                    x = x @ x
                return (x,)

            inp = torch.rand([4096, 4096], device="cuda")
            ref = foo([inp])[0]
            torch.cuda.synchronize()

            user_stream = torch.cuda.Stream()
            with torch.cuda.stream(user_stream):
                foo_cg = self.cudagraphify_impl(foo, [inp], (0,))
                out = foo_cg([inp])[0]
                y = out + 1
                self.assertEqual(y, ref + 1)

        def test_unaligned_static_parameter(self):
            def gen_inp():
                inp = torch.ones([20], device="cuda")
                return [inp[1:]]

            def foo(args):
                x = args[0]
                args.clear()
                return (x + x,)

            foo_cg = self.cudagraphify_impl(foo, gen_inp(), (0,))

            for _ in range(3):
                out = foo_cg(gen_inp())
                self.assertEqual(out, foo(gen_inp()))
                del out

            node = self.curr_node()
            self.assertEqual(node.static_input_data_ptrs, [None])

        def test_amp_cache_disabled(self):
            @torch.compile()
            def foo(x):
                return x + x

            for _ in range(3):
                out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            # amp cache for cudagraph outputs should be disabled
            t2 = torch.rand([4, 4], device="cuda")

            with torch.cuda.amp.autocast():
                run_once = out @ t2

                out.detach().zero_()

                run_twice = out @ t2

                self.assertNotEqual(run_once, run_twice)

        def test_remove_hooks_on_cached_tensors(self):
            @torch.compile()
            def foo(x):
                return x * x

            inp = torch.rand([4], device="cuda", requires_grad=True)

            for _ in range(5):
                out = foo(inp)
                self.assertIsNone(out._backward_hooks)
                out.register_hook(lambda: None)

            # today, torch.compile never outputs a leaf tensor which is the only
            # tensor that can register _post_accumulate_grad_hooks
            # add this as a preventative test

            @torch.compile()
            def foo(x):
                return torch.rand([4], device="cuda", requires_grad=True)

            for _ in range(5):
                out = foo(inp)
                self.assertIsNone(out._post_accumulate_grad_hooks)
                out.register_post_accumulate_grad_hook(lambda: None)

        def test_multiple_insert_removal_caching(self):
            torch._C._set_cached_tensors_enabled(True)
            try:
                x = torch.rand([4], device="cuda")

                torch._C._add_cached_tensor(x)
                self.assertTrue(torch._C._is_cached_tensor(x))

                torch._C._add_cached_tensor(x)
                torch._C._remove_cached_tensor(x)

                self.assertFalse(torch._C._is_cached_tensor(x))
            finally:
                torch._C._set_cached_tensors_enabled(False)

        def test_accumulate_grad(self):
            # cudagraph trees shouldnt interfere with accumulation logic

            def compute_grad(grad_output, create_graph):
                x = torch.randn(5, 5, requires_grad=True, device="cuda")

                @torch.compile()
                def foo(x):
                    return x + 2

                y = foo(x)
                y.backward(grad_output, retain_graph=True)
                x_grad = x.grad
                x_grad_clone = x.grad.clone()
                y.backward(grad_output, create_graph=create_graph)
                return x_grad, x_grad_clone

            for _ in range(3):
                grad_output = torch.ones(5, 5, device="cuda")

                # Accumulate in-place when create_graph is False
                x_grad, x_grad_clone = compute_grad(grad_output, create_graph=False)
                self.assertEqual(x_grad, x_grad_clone * 2)

                # Accumulate out-of-place when create_graph is False
                x_grad, x_grad_clone = compute_grad(grad_output, create_graph=True)
                self.assertEqual(x_grad, x_grad_clone)

        def test_frozen_fn(self):
            @torch.compile()
            def foo(x):
                return x @ x

            for _ in range(3):
                out = foo(torch.rand([10, 10], device="cuda"))

            self.assertTrue(self.get_manager().new_graph_id().id == 1)
            frozen = torch._dynamo.run(foo)

            for _ in range(3):
                out = frozen(torch.rand([10, 10], device="cuda"))

            # didnt do additional recordings
            self.assertTrue(self.get_manager().new_graph_id().id == 2)

        def test_empty_cpu_tensor(self):
            def foo(x):
                return x @ x, torch.tensor([])

            foo_opt = torch.compile(foo)
            x = torch.rand([4], device="cuda")

            for _ in range(3):
                out_opt = foo_opt(x)
                self.assertEqual(foo(x), out_opt)

            self.assertTrue(self.get_manager().new_graph_id().id == 1)

        def test_output_alias(self):
            inp = torch.rand([20, 20], device="cuda")

            def foo(args):
                x = args[0]
                args.clear()
                out = x + x
                return (x, x[0])

            foo_cg = self.cudagraphify_impl(foo, [inp], ())

            for _ in range(3):
                out_1, out_2 = foo_cg([inp])
                self.assertEqual(cdata(out_1), cdata(out_2))
                del out_1, out_2
                self.assertEqual(len(list(self.curr_node().path_live_weakrefs())), 0)

            self.assertEqual(self.curr_node().cached_tensor_outputs, [None, None])

        def test_empty_storage(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return (
                    (x + x + x),
                    torch.zeros([0], device="cuda"),
                    torch.zeros([100], device="cuda")[0:0],
                )

            inp = torch.rand([4], device="cuda")
            for _ in range(3):
                out = foo(inp)
                node = self.curr_node()
                self.assertEqual(len(list(node.path_live_weakrefs())), 1)

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return (x + x + x), torch.rand([4], device="cuda") + 10

            inp = torch.rand([0], device="cuda")
            for _ in range(3):
                out = foo(inp)
                node = self.curr_node()
                self.assertEqual(len(list(node.path_live_weakrefs())), 1)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_aliased_output_checkpoint(self):
            def foo(args):
                x = args[0]
                args.clear()
                y = x + 2
                return x + 1, y, y[0]

            inp = torch.rand([4, 4], device="cuda")
            foo_cg = self.cudagraphify_impl(foo, [inp], ())
            foo_cg([inp])
            foo_cg([inp])

            out1, out2, out3 = foo_cg([inp])
            inp = [out1]

            del out1, out2, out3

            def foo2(args):
                x = args[0]
                args.clear()
                return [x * x * x]

            self.assertEqual(self.num_checkpoints(), 0)
            foo2_cg = self.cudagraphify_impl(foo2, inp, ())

            x = foo2_cg(inp)[0]

            self.assertEqual(self.num_checkpoints(), 1)
            # out2 and out3 dies between the previous recording and the new one,
            # need to be manually deallocated after the checkpoint

            self.assertEqual(all_live_block_count(), 1)
            del x
            self.assertEqual(all_live_block_count(), 0)

        @skipIfRocm
        @unittest.skipUnless(IS_X86 and IS_LINUX, "cpp contexts are linux only")
        @torch._inductor.config.patch("triton.cudagraph_trees_history_recording", True)
        def test_workspace_allocation_error(self):
            torch._C._cuda_clearCublasWorkspaces()

            prev = torch._inductor.cudagraph_trees.clear_cublas_manager

            try:
                torch._inductor.cudagraph_trees.clear_cublas_manager = (
                    contextlib.nullcontext
                )

                @torch.compile()
                def foo(x, y):
                    return x @ x

                inps = [torch.rand([400, 400], device="cuda") for _ in range(2)]

                thrown = False
                try:
                    foo(*inps)
                except Exception as e:
                    thrown = True
                    if not IS_ARM64:
                        self.assertTrue(
                            "at::cuda::blas::gemm<float, float>" in str(e)
                            or "at::cuda::blas::gemm_internal_cublas<float, float>"
                            in str(e)
                        )
                        self.assertTrue(
                            "getCurrentCUDABlasHandle" in str(e)
                            or "getNewWorkspace" in str(e)
                        )

                self.assertTrue(thrown)

            finally:
                torch._C._cuda_clearCublasWorkspaces()
                torch._inductor.cudagraph_trees.clear_cublas_manager = prev
                torch._inductor.cudagraph_trees.get_container(
                    self.device_idx
                ).tree_manager = None

        def test_peristed_output_livenes(self):
            @torch.compile
            def foo(x):
                return x + x

            for _ in range(3):
                foo(torch.rand([2, 2], device="cuda"))

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)

            out = foo(torch.rand([2, 2], device="cuda"))
            self.assertTrue(out is node.cached_tensor_outputs[0])
            self.assertEqual(len(list(node.path_live_weakrefs())), 1)

            out_ref = out[0:]
            del out
            self.assertEqual(len(list(node.path_live_weakrefs())), 1)

            del out_ref
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_no_longer_in_pool(self):
            def foo(args):
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            x1, x2 = foo_cg(inp_list)

            def foo2(args):
                x = args[0]
                args.clear()
                return [x * x * x]

            inp_list = [x1]
            foo2_cg = self.cudagraphify_impl(foo2, inp_list, ())
            foo2_cg(inp_list)

            del x1, x2
            # TODO make configurable

            x1, x2 = foo_cg([inp])
            self.assertEqual(self.num_checkpoints(), 0)

            # input location has changed, should force recompile and checkpointing
            foo2_cg([torch.zeros_like(x1)])

            self.assertEqual(self.num_checkpoints(), 1)
            self.assertEqual(self.get_root_children(), [2])

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_checkpoint_shared_output_storage_deallocation(self):
            def foo(args):
                x = args[0]
                args.clear()
                x_tmp = x + 1
                return x[0], x[1]

            inp = torch.rand([2, 2], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            foo_cg(inp_list)
            foo_cg([inp])

            x1, x2 = foo_cg([inp])
            inp = [x1]

            def foo2(args):
                x = args[0]
                args.clear()
                y = x * x
                return y[0], y[1]

            foo2_cg = self.cudagraphify_impl(foo2, inp, ())
            foo2_cg(inp)

            self.assertEqual(self.num_checkpoints(), 1)
            self.assertEqual(
                x1.untyped_storage().data_ptr(), x2.untyped_storage().data_ptr()
            )
            self.assertEqual(all_live_block_count(), 1)
            del x1
            self.assertEqual(all_live_block_count(), 1)
            del x2
            self.assertEqual(all_live_block_count(), 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_cleanup(self):
            def test_closure():
                @torch.compile
                def foo(x):
                    return x + 1 + 2, x * 10

                foo(torch.rand([4], device="cuda"))
                return foo(torch.rand([4], device="cuda"))

            out1, out2 = test_closure()
            torch._dynamo.reset()

            # TODO - deallocate on tensor deallocation
            # self.assertTrue(self.get_manager() is not None)
            # del out1
            # self.assertTrue(self.get_manager() is not None)
            # del out2
            self.assertTrue(self.get_manager() is None)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_forward_backward(self):
            @torch.compile
            def foo(x):
                y = x * 2
                return torch.sin(y) * torch.nn.functional.dropout(x, p=0.4)

            inp = torch.rand([4, 4], requires_grad=True, device="cuda")
            out = foo(inp)
            out.sum().backward()

            self.assertEqual(self.get_root_children(), [1])

            # the three saved tensors should die in the backward
            # we kept alive the output
            self.assertEqual(self.curr_node().expected_dead_indices_before_graph, [])
            self.assertEqual(
                self.curr_node().expected_dead_indices_after_graph,
                [(0, 1), (0, 2)],
            )
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def test_separate_recordings(self):
            def foo_unopt(x, y):
                return (x + 1) @ y

            foo = torch.compile(foo_unopt)

            foo_unopt(
                torch.ones([20, 20], device="cuda"), torch.ones([20, 20], device="cuda")
            )

            inps = [
                torch.ones([20, 20], device="cuda", requires_grad=False)
                for _ in range(2)
            ]

            out = foo(*inps)
            torch.cuda.synchronize()
            foo(*inps)
            torch.cuda.synchronize()
            foo(*inps)
            torch.cuda.synchronize()

            foo_unopt(
                torch.ones([20, 20], device="cuda"), torch.ones([20, 20], device="cuda")
            )

            inps2 = [
                torch.rand([40, 40], device="cuda", requires_grad=False)
                for _ in range(2)
            ]

            foo(*inps2)
            foo(*inps2)
            foo(*inps2)

            # two separate roots
            self.assertEqual(self.get_root_children(), [0, 0])

        def test_alias_of_parameter(self):
            class AliasMod(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand([20, 20], device="cuda"))

                def forward(self, x):
                    return self.param[0], self.param, self.param + x

            @torch.compile(mode="reduce-overhead")
            def foo(mod, inp):
                return mod(inp)

            inp = torch.rand([20, 20], device="cuda")
            mod = AliasMod()

            storage_ref = torch.multiprocessing.reductions.StorageWeakRef(
                mod.param.untyped_storage()
            )

            for _ in range(3):
                outs = foo(mod, inp)

            self.assertEqual(mod(inp), outs)

            self.assertFalse(storage_ref.expired())

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 1)

        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", False)
        def test_unstable_ptr(self):
            import torch

            @torch.compile(mode="reduce-overhead")
            def foo(m, inp):
                return m(inp)

            def f():
                l = []
                m = torch.nn.Linear(20, 20).cuda()
                for _ in range(4):
                    inp = torch.rand([20, 20], device="cuda")
                    foo(m, inp)
                    m.weight.data = torch.rand([20, 20], device="cuda")

            self.assertRaises(RuntimeError, f)

        @requires_multigpu()
        def test_manager_per_device(self):
            def test():
                def foo(args):
                    x = args[0]
                    args.clear()
                    return (x + 3,)

                inp = torch.rand([20, 20], device="cuda:1")

                inp_list = [inp]
                foo_cg = tree_cudagraphify_impl(
                    foo,
                    inp_list,
                    (),
                    device_index=1,
                    is_backward=False,
                    is_inference=True,
                )
                for _ in range(3):
                    self.assertEqual(foo_cg([inp]), foo([inp]))

                self.assertTrue(self.get_manager(device_index=0) is None)
                self.assertFalse(self.get_manager(device_index=1) is None)

            test()
            self.assertTrue(self.get_manager(device_index=1) is None)

        def test_error_on_dealloc_use(self):
            @torch.compile()
            def foo(x):
                return x * x * x

            inp = torch.rand([4], device="cuda")
            out = foo(inp)
            out2 = foo(inp)

            with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
                out + out

            foo(inp)

            with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
                out2 + out2

        def test_error_on_dealloc_use2(self):
            @torch.compile()
            def foo(x):
                return x * x * x

            inp = torch.rand([4], device="cuda")
            out = foo(inp).detach()
            out2 = foo(inp).detach()

            with self.assertRaises(Exception) as exc:
                out + out

            FileCheck().check("overwritten").check("x * x * x").run(repr(exc.exception))

            foo(inp)

            with self.assertRaises(Exception) as exc:
                out2 + out2

            FileCheck().check("overwritten").check("x * x * x").run(repr(exc.exception))

        @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
        def test_conv_benchmark(self):
            with torch.backends.cudnn.flags(
                enabled=True, benchmark=True, deterministic=False
            ):
                m = torch.nn.Conv2d(5, 6, [3, 3]).cuda()
                inp = torch.randn([2, 5, 16, 16]).cuda()

                @torch.compile()
                def foo(m, inp):
                    return m(inp)

                foo(m, inp)

        def test_single_stream_use(self):
            @torch.compile()
            def foo(x):
                return (x * x * x).relu()

            inp = torch.rand([4], device="cuda", requires_grad=True)
            streams = set()
            streams_init = {seg["stream"] for seg in get_all_cudagraph_segments()}
            for _ in range(4):
                foo(inp).sum().backward()
                inp.grad = None

            streams = {
                seg["stream"] for seg in get_all_cudagraph_segments()
            } - streams_init
            self.assertEqual(len(streams), 1)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._dynamo.config.patch("assume_static_by_default", False)
        def test_dynamic_backward(self):
            def foo(x):
                x = torch.cat([x, x])
                return torch.addmm(x, x, x).relu(), x.size(0)

            opt_foo = torch.compile(mode="reduce-overhead")(foo)

            def run_test(foo, inp):
                r, s = foo(inp)
                r.sum().backward()
                g = inp.grad.clone()
                inp.grad = None
                r = r.clone()
                return r, s, g

            def run_big_test(inp):
                r0, s0, g0 = run_test(foo, inp)
                r1, s1, g1 = run_test(opt_foo, inp)
                r2, s2, g2 = run_test(opt_foo, inp)
                self.assertEqual(r0, r1)
                self.assertEqual(r0, r2)
                self.assertEqual(s0, s1)
                self.assertEqual(s0, s2)
                self.assertEqual(g0, g1)
                self.assertEqual(g0, g2)

            inp = torch.randn(2, 4, device="cuda", requires_grad=True)
            run_big_test(inp)

            inp = torch.randn(3, 6, device="cuda", requires_grad=True)
            run_big_test(inp)

        def test_dynamic_warmup(self):
            COUNTER = 0

            def f(inps):
                i, x = inps
                inps.clear()
                nonlocal COUNTER
                COUNTER += 1
                return x * 2

            x = torch.randn(2, device="cuda")
            inp_list = [2, x]
            foo_cg = self.cudagraphify_impl(f, inp_list, ())
            foo_cg(inp_list)  # warmup
            foo_cg([2, x])  # record
            foo_cg([2, x])  # replay
            self.assertEqual(COUNTER, 2)

            # Switching the size will require a warmup again
            x = torch.randn(3, device="cuda")
            inp_list = [3, x]
            foo_cg(inp_list)  # warmup
            foo_cg([3, x])  # record
            foo_cg([3, x])  # replay
            self.assertEqual(COUNTER, 4)

        def test_forward_generation(self):
            def foo(x):
                return x * x * x

            def foo2(x):
                return x * 12

            foo_opt = torch.compile(foo)
            foo2_opt = torch.compile(foo2)
            ones = torch.ones([4, 4], device="cuda", requires_grad=True)

            out = foo_opt(ones)
            out2 = foo2_opt(out)

            self.assertEqual(all_live_block_count(), 2)

            self.assertTrue(self.get_manager().running_forwards_with_pending_backwards)

            out2.sum().backward()
            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            ones.grad = None
            del out
            del out2

            foo2_opt(foo_opt(ones)).sum().backward()

            out = foo_opt(ones.detach())
            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def test_warn_on_pending_backward(self):
            @torch.compile
            def foo(x):
                return x * x * x

            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            warnings.resetwarnings()
            with warnings.catch_warnings(record=True) as w:
                out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            FileCheck().check(
                "Unable to hit fast path of CUDAGraphs because of pending"
            ).run(str(w[0]))
            self.assertTrue(self.get_manager().new_graph_id().id == 0)

        def test_mark_step(self):
            @torch.compile
            def foo(x):
                return x * x * x

            torch.compiler.cudagraph_mark_step_begin()
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            torch.compiler.cudagraph_mark_step_begin()
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._dynamo.config.patch("capture_scalar_outputs", True)
        def test_incompatible_cudagraph_ops_item(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x.item()

            # NB: This doesn't work with float, because float unbacked codegen
            # is currently broken.  But testing the float case here is also
            # awkward, because we plan to Tensor-ify the float compute, and as
            # a result we'd actually expect this to work with cuda graphs!
            with capture_stderr() as captured_output:
                self.assertEqual(foo(torch.tensor(3, device="cuda")), 3)
                self.assertEqual(foo(torch.tensor(6, device="cuda")), 6)

            # NOTE: this test is named after incompatible ops, but is not skipping due to incompatible ops.
            # This should get fixed.
            FileCheck().check(
                " to incompatible op aten._local_scalar_dense.default"
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @torch._dynamo.config.patch("compiled_autograd", True)
        def test_compiled_autograd_static_input_params(self):
            @torch.compile(mode="reduce-overhead")
            def bwd(loss):
                loss.backward()

            model = torch.nn.Linear(10, 10, bias=False, device="cuda")
            x = torch.randn(10, 10, device="cuda")
            for i in range(5):
                out = model(x)
                bwd(out.sum())
                model.weight.grad = None

            # i=0, 0 copies (warmup)
            # i=1, 2 copies (record, 1/3 inputs marked as static)
            # i>1, 0 copies (run)
            self.assertEqual(
                counters["inductor"]["cudagraph_recorded_non_static_inputs"], 2
            )

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_incompatible_cudagraph_ops_nonzero(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x.nonzero()

            with capture_stderr() as captured_output:
                self.assertEqual(
                    foo(torch.tensor([1, 0, 2], device="cuda")),
                    torch.tensor([[0], [2]]),
                )
                self.assertEqual(
                    foo(torch.tensor([1, 0, 0], device="cuda")), torch.tensor([[0]])
                )

            FileCheck().check("incompatible op aten.nonzero.default").check("foo").run(
                captured_output[0]
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_incompatible_cudagraph_ops_nonzero_graph_breaks(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                y = x.nonzero()  # skip
                torch._dynamo.graph_break()
                return y.nonzero()  # skip 2 times (due to recompile)

            foo(torch.tensor([1, 0, 2], device="cuda"))
            foo(torch.tensor([1, 0, 0], device="cuda"))

            self.assertEqual(counters["inductor"]["cudagraph_skips"], 3)

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_incompatible_cudagraph_ops_nonzero_backend(self):
            @torch.compile(backend="cudagraphs")
            def foo(x):
                return x.nonzero()

            with capture_stderr() as captured_output:
                self.assertEqual(
                    foo(torch.tensor([1, 0, 2], device="cuda")),
                    torch.tensor([[0], [2]]),
                )
                self.assertEqual(
                    foo(torch.tensor([1, 0, 0], device="cuda")), torch.tensor([[0]])
                )

            FileCheck().check(
                "skipping cudagraphs due to incompatible op (nonzero)"
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        @torch._inductor.config.patch("cpp_wrapper", True)
        def test_skip_cpp_wrapper(self):
            def foo(x):
                return x + 1

            foo_c = torch.compile(mode="reduce-overhead")(foo)

            with capture_stderr() as captured_output:
                t = torch.rand([32], device="cuda")
                self.assertEqual(foo(t), foo_c(t))

            FileCheck().check("skipping cudagraphs due to cpp wrapper enabled").run(
                captured_output[0]
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        def test_storage_access_error(self):
            x = torch.rand([4], device="cuda")
            torch._C._set_storage_access_error_msg(x, "custom error msg")

            with self.assertRaisesRegex(Exception, "custom error msg"):
                device = x.untyped_storage()

        def test_side_stream_memory_allocation(self):
            from torch._inductor.cudagraph_trees import cudagraphify_impl

            def multi_stream_allocation(args):
                side_stream = torch.cuda.Stream()
                side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_stream):
                    side_stream_buffer = torch.ones(
                        *args, device="cuda:0", dtype=torch.float32
                    )
                torch.cuda.current_stream().wait_stream(side_stream)

                main_stream_buffer = torch.ones(
                    *args, device="cuda:0", dtype=torch.float32
                )

                if isinstance(args, list):
                    args.clear()

                return main_stream_buffer, side_stream_buffer

            graphed_multi_stream_func = cudagraphify_impl(
                multi_stream_allocation,
                inputs=[],
                static_input_idxs=[],
                is_backward=False,
                is_inference=False,
                device_index=0,
                stack_traces=["dummy stack trace1", "dummy stack trace2"],
            )

            ref_out = torch.ones((2, 3), device="cuda:0", dtype=torch.float32)

            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                main_stream_buffer, side_stream_buffer = graphed_multi_stream_func(
                    [2, 3]
                )
                self.assertEqual(main_stream_buffer, ref_out)
                self.assertEqual(side_stream_buffer, ref_out)

            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", False)
        def test_static_inputs_address_mutation_log(self):
            class Goo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(2, 2, device="cuda")

                def forward(self, x) -> torch.Tensor:
                    return self.linear(x)

            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.static_tensor = torch.zeros((2, 2), device="cuda")
                    self.goo = Goo()

                def forward(self, x) -> torch.Tensor:
                    self.static_tensor.add_(torch.ones((2, 2), device="cuda"))
                    return self.static_tensor + x + self.goo(x)

            foo = Foo()
            foo = torch.compile(foo, mode="reduce-overhead")
            inp = torch.rand((2, 2), device="cuda")

            for _ in range(3):
                foo(inp)

            # mutates static input tensors' addresses
            foo.static_tensor = torch.ones((2, 2), device="cuda")
            foo.goo.linear.bias = torch.nn.Parameter(torch.ones((2,), device="cuda"))

            with self.assertRaisesRegex(
                Exception,
                r"(?s)static input data pointer changed.\n"
                r"input name: primals_2. data pointer changed from .* to .*. input stack trace:.*"
                r"input name: primals_3. data pointer changed from .* to .*. input stack trace:.*,"
                r" in forward\n.* self.static_tensor.add\_\(torch.ones\(\(2, 2\), device=\"cuda\"\)\).*\n",
            ):
                self.curr_node().run(
                    [foo.goo.linear.weight, foo.goo.linear.bias, foo.static_tensor, inp]
                )

        def _run_iter(self, param, fn):
            fwd_output = fn(torch.ones(2, 2), param)
            fwd_output.sum().backward()
            grad_output = param.grad.detach().clone()
            param.grad = None
            return fwd_output, grad_output

        def _assert_equal_multi_loop(self, param, fn_eager, fn_compiled):
            exp_output, exp_grad = self._run_iter(param, fn_eager)
            for _ in range(5):
                compiled_output, compiled_grad = self._run_iter(param, fn_compiled)
                self.assertEqual(exp_output, compiled_output)
                self.assertEqual(exp_grad, compiled_grad)

        def run_static_input_param_test(self, fn_eager, num_graphs):
            with torch.device("cuda"):
                fn_compiled = torch.compile(fn_eager, mode="reduce-overhead")

                p1 = torch.nn.Parameter(torch.rand([2, 2]))
                self._assert_equal_multi_loop(p1, fn_eager, fn_compiled)

                p2 = torch.nn.Parameter(torch.rand([2, 2]))
                self._assert_equal_multi_loop(p2, fn_eager, fn_compiled)

                # Run p1 again to ensure we reuse the previous recording
                self._assert_equal_multi_loop(p1, fn_eager, fn_compiled)

                self.assertEqual(self.get_manager().new_graph_id().id, num_graphs)

        def _module_test(self, mod, name="weight", param_wrapping=True):
            with torch.device("cuda"):

                def fn(x, mod):
                    return mod(x)

                fn_compiled = torch.compile(fn, mode="reduce-overhead", fullgraph=True)

                def run_test_iter(mod, fn):
                    fwd_output = fn(torch.ones(2, 2), mod)
                    fwd_output.sum().backward()
                    grad_output = mod.weight.grad.detach().clone()
                    mod.zero_grad()
                    return fwd_output, grad_output

                def run_test():
                    exp_output, exp_grad = run_test_iter(mod, fn)
                    for _ in range(5):
                        compiled_output, compiled_grad = run_test_iter(mod, fn_compiled)
                        self.assertEqual(exp_output, compiled_output)
                        self.assertEqual(exp_grad, compiled_grad)

                run_test()
                old_attr = getattr(mod, name)
                modified_attr = torch.rand_like(old_attr)
                if param_wrapping:
                    modified_attr = torch.nn.Parameter(modified_attr)
                setattr(mod, name, modified_attr)
                run_test()
                # Run original version to verify we reuse the other recording
                setattr(mod, name, old_attr)
                run_test()

                # Fwd + bwd graphs for each version of the function => 4 graphs
                self.assertEqual(self.get_manager().new_graph_id().id, 4)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_single_compile_param_inputs(self):
            # Verify that we can record multiple cudagraphs for a single
            # compiled function with param inputs
            def fn(x, y):
                return x * y

            # Fwd + bwd graphs for each version of the function => 4 graphs
            self.run_static_input_param_test(fn, 4)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_single_compile_builtin_module(self):
            # Verify that we don't recompile when changing the param of a builtin module
            # and that we record another cudagraph
            # Note: Linear is a builtin module so we enable that config setting above
            self._module_test(torch.nn.Linear(2, 3, device="cuda"))

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_single_compile_builtin_module_buffers(self):
            # Verify that we don't recompile when changing the buffer of a builtin module
            # and that we record another cudagraph
            self._module_test(
                torch.nn.BatchNorm1d(2, device="cuda"),
                name="running_mean",
                param_wrapping=False,
            )

        @torch._inductor.config.patch("triton.cudagraphs", True)
        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_custom_module(self):
            # Test that we can correctly dispatch multiple graphs
            # if params of a custom module change
            class TestModule(torch.nn.Module):
                def __init__(self, param) -> None:
                    super().__init__()
                    self.weight = param

                def forward(self, x):
                    return self.weight * x

            self._module_test(
                TestModule(torch.nn.Parameter(torch.rand([2, 2], device="cuda")))
            )

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_custom_module_buffer(self):
            # Test that we can correctly dispatch multiple graphs
            # if buffers of a custom module change
            class TestModule(torch.nn.Module):
                def __init__(self, param, buf) -> None:
                    super().__init__()
                    self.weight = param
                    self.buf = torch.nn.Buffer(buf)

                def forward(self, x):
                    return x * self.weight + self.buf

            self._module_test(
                TestModule(
                    torch.nn.Parameter(torch.rand([2, 2], device="cuda")),
                    torch.rand([2, 2], device="cuda"),
                ),
                name="buf",
                param_wrapping=False,
            )

        @torch._inductor.config.patch("triton.cudagraphs", True)
        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_child_node(self):
            # Test that we can correctly dispatch multiple graphs if a child node
            # in the tree has stable input pointers change
            def fn(x, p):
                # Graph 1
                y = x * x
                torch._dynamo.graph_break()
                # Graph 2
                return y * p

            # We have 5 graphs here
            #            Graph 1
            #       /                \
            # Graph 2 w/ p1     Graph 2 w/ p2
            # and then two backward graphs
            self.run_static_input_param_test(fn, 5)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_multi_dispatch_parent_node(self):
            def fn(x, p):
                # Graph 1
                y = x * p
                torch._dynamo.graph_break()
                # Graph 2
                return y + x

            # We have 6 graphs here
            #    Graph 1 w/ p1    Graph 1 w/ p2
            #          |                |
            #     Graph 2 (v1)     Graph 2 (v2)
            # There are two versions of graph 2 because
            # we re-record due to different memory state after running the
            # two versions of Graph 1
            # and then two backward graphs
            self.run_static_input_param_test(fn, 6)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 0)
        def test_fallback_to_eager_if_recompiling_too_many_times(self):
            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand([2, 2], device="cuda"))

                def forward(self, x):
                    return x * self.param

            with capture_stderr() as captured_output:
                # We have 3 graphs here
                #             None
                #       /                           \
                # (fwd w/ p1, Graph 0)            (bwd w/p2, Graph2)
                # (bwd w/ p1, Graph 1)
                # All other graphs are skipped because we hit the max recording limit
                # (=0 for each node and function pair)
                fn_compiled = torch.compile(Foo(), mode="reduce-overhead")
                for _ in range(3):
                    fn_compiled(torch.rand([2, 2], device="cuda")).sum().backward()
                    fn_compiled.param.grad = None

                # Change static tensor address
                fn_compiled.param.data = torch.rand([2, 2], device="cuda")
                fn_compiled(torch.rand([2, 2], device="cuda")).sum().backward()
                self.assertEqual(self.get_manager().new_graph_id().id, 3)

            FileCheck().check(
                "skipping cudagraph due to function 0 exceeding max re-recording limit (=0) "
                "on cudagraph node None due to static input data pointer changed."
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 0)
        def test_fallback_to_eager_if_recompiling_too_many_times_warn_only_once(self):
            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand([2, 2], device="cuda"))

                def forward(self, x):
                    return x * self.param

            with capture_stderr() as captured_output:
                with torch.device("cuda"):
                    # We have 3 graphs here
                    #             None
                    #       /                           \
                    # (fwd w/ p1, Graph 0)            (bwd w/p2, Graph2)
                    # (bwd w/ p1, Graph 1)
                    # All other graphs are skipped because we hit the max recording limit
                    # (=0 for each node and function pair)
                    fn_compiled = torch.compile(Foo(), mode="reduce-overhead")
                    for _ in range(3):
                        fn_compiled(torch.rand([2, 2], device="cuda")).sum().backward()
                        fn_compiled.param.grad = None

                    for _ in range(5):
                        # Change static tensor address
                        fn_compiled.param.data = torch.rand([2, 2], device="cuda")
                        fn_compiled(torch.rand([2, 2], device="cuda")).sum().backward()
                        fn_compiled.param.grad = None

            FileCheck().check_count(
                "skipping cudagraph due to function 0 exceeding max re-recording limit (=0) "
                "on cudagraph node None due to static input data pointer changed.",
                1,
                exactly=True,
            ).check_count(
                "skipping cudagraph due to function 1 exceeding max re-recording limit (=0) "
                "on cudagraph node None due to static input data pointer changed.",
                1,
                exactly=True,
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 2)

        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 0)
        def test_fallback_to_eager_if_recompiling_too_many_times_due_to_cudagraph_managed_tensor(
            self,
        ):
            # By setting triton.cudagraph_support_input_mutation=True, we force re-record
            # if cudagraph managed tensor addresses changed.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x + 1

            @torch.compile(mode="reduce-overhead")
            def goo(x):
                return x * 2

            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                inp = torch.rand((2, 3), device="cuda")
                y = foo(inp)
                z = goo(y)

            with capture_stderr() as captured_output:
                torch.compiler.cudagraph_mark_step_begin()
                x = torch.rand(2, 3, device="cuda")
                y = foo(x)
                y_clone = y.clone()
                z = goo(y_clone)

            # eager function should run successfully
            for _ in range(5):
                torch.compiler.cudagraph_mark_step_begin()
                x = torch.rand(2, 3, device="cuda")
                y = foo(x)
                y_clone = y.clone()
                z = goo(y_clone)

            FileCheck().check_count(
                "skipping cudagraph due to function 1 exceeding max re-recording limit (=0) "
                "on cudagraph node 0 due to cudagraph managed tensor data pointer changed",
                1,
                exactly=True,
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", False)
        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 1)
        def test_not_fallback_to_eager_if_have_not_recompiling_too_many_times(self):
            def fn(x, y):
                return x * y

            # We have 4 graphs here
            #             None
            #       /                           \
            # (fwd w/ p1, Graph 0)            (fwd w/p2, Graph2)
            # (bwd w/ p1, Graph 1)            (bwd w/p2, Graph3)
            self.run_static_input_param_test(fn, 4)
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
        def test_no_rerecord_with_mark_static_address(self):
            class Mod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(2, 2)

                def forward(self, x):
                    return self.linear(x)

            mod = Mod().cuda()

            def fn_eager(x, marked_static_y):
                return torch.cos(x) + mod(marked_static_y)

            with torch.device("cuda"):
                fn_compiled = torch.compile(fn_eager, mode="reduce-overhead")

                # y is marked static
                y = torch.randn(2, 2)
                torch._dynamo.mark_static_address(y)

                # Chanhing pointer of x should not lead to re-records
                for _ in range(5):
                    x = torch.randn(2, 2, requires_grad=True)
                    res = fn_compiled(x, y)
                    res.sum().backward()
                    x.grad = None
                    mod.linear.weight.grad = None
                    mod.linear.bias.grad = None
                # One forward and one backward
                self.assertEqual(self.get_manager().new_graph_id().id, 2)

        def test_tensor_constant_mutation(self):
            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.tensor_constant = torch.ones((2, 3), device="cuda")

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.tensor_constant += 1
                    return x + self.tensor_constant

            foo = Foo()
            foo = torch.compile(foo, mode="reduce-overhead")
            inp = torch.rand((2, 3), device="cuda")
            for _ in range(3):
                foo(inp)

        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_rerecord_if_static_input_address_changed(self):
            # By setting triton.cudagraph_support_input_mutation=True, we force re-record
            # if static tensor addresses changed.
            class Goo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(2, 2, device="cuda")

                def forward(self, x) -> torch.Tensor:
                    return self.linear(x)

            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.register_buffer(
                        "static_tensor", torch.zeros((2, 2), device="cuda")
                    )
                    self.goo = Goo()

                def forward(self, x) -> torch.Tensor:
                    self.static_tensor.add_(torch.ones((2, 2), device="cuda"))
                    return self.static_tensor + x + self.goo(x)

            foo = Foo()
            foo = torch.compile(foo, mode="reduce-overhead")
            inp = torch.rand((2, 2), device="cuda")

            for _ in range(3):
                foo(inp)

            # mutates static input tensors' addresses
            foo.static_tensor = torch.ones((2, 2), device="cuda")
            foo.goo.linear.bias = torch.nn.Parameter(torch.ones((2,), device="cuda"))

            if torch._dynamo.config.inline_inbuilt_nn_modules:
                for _ in range(3):
                    foo(inp)
            else:
                # Run with specific function id to avoid dynamo recompiling
                self.get_manager().run(
                    [
                        foo.goo.linear.weight,
                        foo.goo.linear.bias,
                        foo.static_tensor,
                        inp,
                    ],
                    FunctionID(0),
                )

            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("triton.cudagraph_dynamic_shape_warn_limit", 1)
        def test_skip_if_dynamic_shape_limit_reached1(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(3, 3, device="cuda")

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.linear(x)

            def iter(batch_size: int, mod: torch.nn.Module):
                x = torch.rand((batch_size, 3), device="cuda")
                for _ in range(3):
                    mod(x)

            mod = torch.compile(Mod(), mode="reduce-overhead")

            with capture_stderr() as captured_output:
                for batch_size in range(10, 40, 10):
                    iter(batch_size, mod)

            FileCheck().check(
                "CUDAGraph supports dynamic shapes by recording a new graph for each "
                "distinct input size. Recording too many CUDAGraphs may lead to "
                "extra overhead. We have observed 2 distinct sizes. "
                "Please consider the following options for better performance: "
                "a) padding inputs to a few fixed number of shapes; or b) set "
                "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
                "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
                "to silence this warning."
            ).run("\n".join(captured_output))

        @torch._inductor.config.patch("triton.cudagraph_dynamic_shape_warn_limit", 1)
        def test_skip_if_dynamic_shape_limit_reached2(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.attn = torch.nn.MultiheadAttention(
                        embed_dim=3, num_heads=3, device="cuda"
                    )

                def forward(
                    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
                    return self.attn(q, k, v)

            mod = torch.compile(Mod(), mode="reduce-overhead")

            def iter(batch_size: int, length: int):
                q = torch.rand((batch_size, length, 3), device="cuda")
                k = torch.rand((batch_size, length, 3), device="cuda")
                v = torch.rand((batch_size, length, 3), device="cuda")
                for _ in range(3):
                    mod(q, k, v)

            with capture_stderr() as captured_output:
                for batch_size in range(10, 40, 10):
                    for length in range(10, 30, 10):
                        iter(batch_size, length)

            print(captured_output)
            FileCheck().check(
                "CUDAGraph supports dynamic shapes by recording a new graph for each "
                "distinct input size. Recording too many CUDAGraphs may lead to "
                "extra overhead. We have observed 2 distinct sizes. "
                "Please consider the following options for better performance: "
                "a) padding inputs to a few fixed number of shapes; or b) set "
                "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
                "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
                "to silence this warning."
            ).run(captured_output[0])

        @torch._inductor.config.patch("triton.cudagraph_dynamic_shape_warn_limit", 1)
        def test_warn_once_if_dynamic_shape_limit_reached(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(3, 3, device="cuda")

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.linear(x)

            def iter(batch_size: int, mod: torch.nn.Module):
                x = torch.rand((batch_size, 3), device="cuda")
                for _ in range(3):
                    mod(x)

            mod = torch.compile(Mod(), mode="reduce-overhead")

            with capture_stderr() as captured_output:
                for batch_size in range(10, 200, 10):
                    iter(batch_size, mod)

            print(captured_output)

            FileCheck().check_count(
                "CUDAGraph supports dynamic shapes by recording a new graph for each "
                "distinct input size. Recording too many CUDAGraphs may lead to "
                "extra overhead. We have observed 2 distinct sizes. "
                "Please consider the following options for better performance: "
                "a) padding inputs to a few fixed number of shapes; or b) set "
                "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
                "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
                "to silence this warning.",
                1,
                exactly=True,
            ).run("\n".join(captured_output))

        @torch._inductor.config.patch("cpp_wrapper", 1)
        def test_cpp_wrapper(self):
            def f(x):
                return torch.sin(x)

            compiled = torch.compile(f, mode="reduce-overhead")
            example_input = torch.randn(10, device="cuda")
            compiled_result = self.run_twc(compiled, example_input)
            eager_result = f(example_input)
            self.assertEqual(compiled_result, eager_result)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition(self):
            def f(x, y):
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.cuda()

            x, y = [torch.randn(2, 2, device="cuda") for _ in range(2)]
            x_cloned, y_cloned = [tmp.clone() for tmp in [x, y]]
            eager_out = f(x, y)

            f_compiled = torch.compile(f, mode="reduce-overhead")

            for _ in range(5):
                compiled_out = f_compiled(x_cloned, y_cloned)
                self.assertEqual(eager_out, compiled_out)

            # 2 graph partitions lead to 2 cudagraph
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar1(self):
            def f(x, y):
                return x + y

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (torch.ones(2, 2, device="cuda"), torch.ones((), device="cpu"))
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar2(self):
            def f(x, y, z):
                return x + y, x + z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.ones((), device="cpu"),
                torch.ones(2, 2, device="cuda"),
                torch.ones(2, 2, device="cuda"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar3(self):
            def f(x, y, cpu_scalar_tensor):
                z = x + y
                z = z + cpu_scalar_tensor
                return z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.randn(2, 2, device="cuda"),
                torch.randn(2, 2, device="cuda"),
                torch.tensor(1, device="cpu"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar4(self):
            # cpu_scalar_tensor is accessed by cpu_scalar2 which is
            # added with a gpu tensor z. This test checks the cpu
            # scalar tensors are still moved in this case.
            def f(x, y, cpu_scalar_tensor):
                cpu_scalar2 = cpu_scalar_tensor + 1
                z = x + y
                z = z + cpu_scalar2
                return z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.randn(2, 2, device="cuda"),
                torch.randn(2, 2, device="cuda"),
                torch.tensor(1, device="cpu"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        # turn on input mutation support to avoid skipping cudagraph at dynamo level
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_graph_partition_cpu_scalar_mutation(self):
            # tests that input mutation on a cpu scalar tensor x is correctly
            # handled when moving x to gpu at the beginning of the graph.

            @torch.compile(mode="reduce-overhead")
            def foo(x, y):
                return x.copy_(y)

            x = torch.tensor(1)
            y = torch.tensor(2, device="cuda")

            for _ in range(3):
                foo(x, y)

            self.assertEqual(x, torch.tensor(2, device="cpu"))
            self.assertEqual(y, torch.tensor(2, device="cuda"))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar_device_put(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                y = x.to("cuda")
                z = y.to("cpu")
                return z

            x = torch.tensor(1)
            for _ in range(3):
                foo(x)

            self.assertEqual(x, torch.tensor(1, device="cpu"))

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_graph_partition_reduce_overhead_mode_effectiveness(self):
            # test that `mode="reduce-overhead"` still controls whether
            # cudagraph is applied. i.e., cudagraph is not applied when
            # mode="default".
            def f(x, y):
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.cuda()

            x, y = [torch.randn(2, 2, device="cuda") for _ in range(2)]

            f_compiled = torch.compile(f)
            for _ in range(5):
                _out = f_compiled(x, y)
            self.assertEqual(self.get_manager() is None, True)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_forward_backward(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(16, 16)

                def forward(self, x):
                    x1 = x + 1
                    y1 = x + 2
                    y_cpu = y1.cpu() + 1
                    z = x @ y1
                    inp = x1 + y1 + z + y_cpu.cuda()
                    return self.linear(inp)

            model = Mod().cuda()

            input_data = torch.randn(16, 16).cuda()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            compiled_model = torch.compile(model, mode="reduce-overhead")

            for _ in range(5):
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 2 graph partitions lead to 2 fwd cudagraphs and 1 bwd cudagraphs
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_only(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(16, 16)

                def forward(self, x):
                    x1 = x + 1
                    y1 = x + 2
                    y_cpu = y1 + 1
                    z = x @ y1
                    inp = x1 + y1 + z + y_cpu
                    return self.linear(inp)

            model = Mod().cpu()

            input_data = torch.randn(16, 16).cpu()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            compiled_model = torch.compile(model, mode="default")

            for _ in range(5):
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 0 cudagraph since all ops are on cpu
            self.assertEqual(self.get_manager() is None, True)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_forward_with_skipped_cudagraphed_backward(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            for _ in range(3):
                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)

                with config.patch(always_complex_memory_overlap_TESTING_ONLY=True):
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # we should not have cudagraph'd the backwards
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_forward_backward_not_called(self):
            # tests saved tensor is handled correctly
            def foo(x, y):
                x_out = x * x * x
                torch._dynamo.graph_break()
                y_out = y * y * y
                return x_out, y_out

            foo = torch.compile(foo, mode="reduce-overhead")

            for _ in range(3):
                inps = [
                    torch.rand([20, 20], requires_grad=True, device="cuda")
                    for _ in range(2)
                ]
                x_out, y_out = foo(inps[0], inps[1])
                x_out.sum().backward()

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            # we should not have cudagraph'd the y backward
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 3)

        @requires_multigpu()
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_multiple_devices_msg(self):
            def foo(x, y):
                return (x + 1, y + 2)

            foo = torch.compile(foo, mode="reduce-overhead")
            for _ in range(3):
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

            with capture_stderr() as captured_output:
                for _ in range(3):
                    foo(
                        torch.ones([10], device="cuda:0"),
                        torch.ones([10], device="cuda:1"),
                    )

            FileCheck().check("skipping cudagraphs due to multiple devices").run(
                captured_output[0]
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_dynamic_shapes(self):
            def foo(x):
                return x + 1

            compiled_foo = torch.compile(foo, mode="reduce-overhead", fullgraph=True)

            for input_shape in range(1, 4):
                for _ in range(3):
                    compiled_foo(torch.randn(input_shape, device="cuda"))

            # 3 cudagraphs for 3 input shapes
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_op_and_dynamic_shapes(self):
            def f(x, y):
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.cuda()

            f_compiled = torch.compile(f)
            x, y = torch.ones(3, 3, device="cuda"), torch.randn(3, 3, device="cuda")
            for _ in range(3):
                compiled_out = f_compiled(x, y)
                self.assertEqual(compiled_out, f(x, y))

            x, y = torch.ones(4, 4, device="cuda"), torch.randn(4, 4, device="cuda")
            for _ in range(3):
                compiled_out = f_compiled(x, y)
                self.assertEqual(compiled_out, f(x, y))

            # 4 cudagraphs, due to (2 dynamic shapes) x (2 graph partitions)
            self.assertEqual(self.get_manager().new_graph_id().id, 4)

        @config.patch(implicit_fallbacks=True)
        @config.patch("graph_partition", False)
        def test_skip_cudagraph_unsafe_ops(self):
            @torch.library.custom_op(
                "mylib::mysin",
                mutates_args=["out_list"],
                schema="(Tensor x, Tensor(a!)[]? out_list) -> Tensor",
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def mysin(x, out_list) -> torch.Tensor:
                r = x.sin()
                if out_list is not None:
                    out_list[0].copy_(r)
                return r

            @mysin.register_fake
            def _(x, out_list) -> torch.Tensor:
                return torch.empty_like(x)

            def fn(x):
                x = x * 3
                s = [torch.empty_like(x)]
                x = mysin(x, s)
                x = x / 3
                return x, s[0]

            x = torch.randn(3, requires_grad=False, device="cuda")
            expected = fn(x)
            compiled_f = torch.compile(fn, mode="reduce-overhead", fullgraph=True)

            with capture_stderr() as captured_output:
                for _ in range(3):
                    result = compiled_f(x)
                    self.assertEqual(result, expected)

            FileCheck().check("incompatible op mylib.mysin.default").run(
                captured_output[0]
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op(self):
            @torch.library.custom_op(
                "mylib::movement",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def movement(pic: torch.Tensor) -> torch.Tensor:
                img = pic.cpu()
                cropped_img = (img + 1) * 2
                return cropped_img.cuda() / 255.0

            @movement.register_fake
            def _(pic):
                return torch.empty_like(pic)

            @torch.library.custom_op(
                "mylib::modify",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def modify(pic: torch.Tensor) -> torch.Tensor:
                pic1 = pic + 1
                pic1_cpu = (pic1.cpu() + 1) * 2
                return pic1_cpu.cuda() + pic

            @modify.register_fake
            def _(pic):
                return torch.empty_like(pic)

            @torch.library.custom_op("mylib::transform", mutates_args=())
            def transform(pic: torch.Tensor) -> torch.Tensor:
                return (pic + 1) * 2

            @transform.register_fake
            def _(pic):
                return torch.empty_like(pic)

            img = torch.randn(3, 64, 64, device="cuda")

            def f(img):
                x = (img + 10) * 2
                y = movement(x)
                z = y + 1
                u = transform(z)
                v = 2 * u + 1
                out = modify(v)
                return out + 1

            compiled_f = torch.compile(f, fullgraph=True)

            eager_out = f(img)
            compiled_out = compiled_f(img)

            self.assertEqual(eager_out, compiled_out)

            compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

            eager_out = f(img)

            for _ in range(3):
                compiled_out = compiled_f(img)
                self.assertEqual(eager_out, compiled_out)

            # splitting on 2 custom gives 3 cudagraphs
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @config.patch(implicit_fallbacks=True)
        @config.patch("graph_partition", True)
        def test_graph_partition_custom_op_mutation(self):
            @torch.library.custom_op(
                "mylib::mysin",
                mutates_args=["out_list"],
                schema="(Tensor x, Tensor(a!)[]? out_list) -> Tensor",
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def mysin(x, out_list) -> torch.Tensor:
                r = x.sin()
                if out_list is not None:
                    out_list[0].copy_(r)
                return r

            @mysin.register_fake
            def _(x, out_list) -> torch.Tensor:
                return torch.empty_like(x)

            def fn(x):
                x = x * 3
                s = [torch.empty_like(x)]
                x = mysin(x, s)
                x = x / 3
                return x, s[0]

            x = torch.randn(3, requires_grad=False, device="cuda")
            expected = fn(x)
            compiled_f = torch.compile(fn, mode="reduce-overhead", fullgraph=True)
            for _ in range(3):
                result = compiled_f(x)
                self.assertEqual(result, expected)

            # splitting on 1 custom gives 2 cudagraphs
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op_dynamoc_shapes(self):
            @torch.library.custom_op(
                "mylib::movement",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def movement(pic: torch.Tensor) -> torch.Tensor:
                img = pic.cpu()
                cropped_img = (img + 1) * 2
                return cropped_img.cuda() / 255.0

            @movement.register_fake
            def _(pic):
                return torch.empty_like(pic)

            def f(img):
                x = (img + 10) * 2
                y = movement(x)
                z = y + 1
                v = 2 * z + 1
                return v + 1

            compiled_f = torch.compile(f, fullgraph=True)

            compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

            def run(size):
                img = torch.randn(3, size, size, device="cuda")
                eager_out = f(img)
                for _ in range(3):
                    compiled_out = compiled_f(img)
                    self.assertEqual(eager_out, compiled_out)

            run(64)
            run(17)
            run(42)

            # 2 (from splitting on 1 custom op) x 3 (dynamic shapes) = 6
            self.assertEqual(self.get_manager().new_graph_id().id, 6)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op_no_split(self):
            @torch.library.custom_op(
                "mylib::modify",
                mutates_args=(),
            )
            def modify(x: torch.Tensor) -> torch.Tensor:
                return (x + 1) * 2

            @modify.register_fake
            def _(pic):
                return torch.empty_like(pic)

            def f(img):
                x = (img + 10) * 2
                y = modify(x)
                z = y + 1
                v = 2 * z + 1
                return v + 1

            compiled_f = torch.compile(f, fullgraph=True)

            compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

            def run(size):
                img = torch.randn(3, size, size, device="cuda")
                eager_out = f(img)
                for _ in range(3):
                    compiled_out = compiled_f(img)
                    self.assertEqual(eager_out, compiled_out)

            run(64)
            run(17)
            run(42)

            # 1 (from not splitting on custom op) x 3 (dynamic shapes) = 3
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_tensor_symints(self):
            def f(x, y):
                return x + 1, y + 1

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(shape_x, shape_y):
                x = torch.randn(shape_x, device="cuda")
                y = torch.randn(shape_y, device="cpu")
                for _ in range(3):
                    compiled_f(x, y)

            # static shape. record a NEW cudagraph
            run(shape_x=(2, 3), shape_y=(4, 4))

            # shape_y becomes dynamic shape leading to a new dynamo graph.
            # This new dynamo graph forces a NEW cudagraph although tensor y is on cpu
            run(shape_x=(2, 3), shape_y=(5, 6))

            # tensor y is on cpu so NO new cudagraph is recorded
            run(shape_x=(2, 3), shape_y=(7, 8))

            # shape_x becomes dynamic shape, leading to a new dynamo graph
            # this new dynamo graph forces a NEW cudagraph
            run(shape_x=(3, 4), shape_y=(4, 4))

            # tensor y is on cpu so NO new cudagraph is recorded
            run(shape_x=(3, 4), shape_y=(10, 11))

            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_cpu_and_gpu(self):
            def f(x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu):
                x_cuda0 = x_cuda + 1
                x_cuda1 = x_cuda0 @ weight_cuda
                x_cuda2 = 2 * (x_cuda1 + x_cuda)

                y_cpu0 = y_cpu + 1
                y_cpu1 = y_cpu0 @ weight_cpu

                z_cuda0 = z_cuda + 1
                z_cuda1 = z_cuda0 @ weight_cuda
                z_cuda2 = 2 * (z_cuda1 + z_cuda)

                return x_cuda2, y_cpu1, z_cuda2

            x_cuda = torch.randn(3, 3, device="cuda")
            y_cpu = torch.randn(3, 3, device="cpu")
            z_cuda = torch.randn(3, 3, device="cuda")
            weight_cuda = torch.randn(3, 3, device="cuda")
            weight_cpu = torch.randn(3, 3, device="cpu")

            eager_out = f(x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu)

            compiled_f = torch.compile(f, mode="reduce-overhead")
            for _ in range(3):
                compiled_out = compiled_f(
                    x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu
                )
                self.assertEqual(eager_out, compiled_out)

            # reorder merges ops on cuda into 1 graph partition
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_cpu_and_gpu_interleave(self):
            def f(x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu):
                # partition 1 on cuda, no dependency
                x_cuda0 = x_cuda + 1
                x_cuda1 = x_cuda0 @ weight_cuda
                x_cuda2 = 2 * (x_cuda1 + x_cuda)

                # partition 2 on cpu w/ dependency on partition 1
                y_cpu0 = y_cpu + 1
                x_cuda2_cpu = x_cuda2.cpu()  # adds dependency on gpu computations
                y_cpu1 = y_cpu0 @ weight_cpu + x_cuda2_cpu

                # partition 3 on cuda w/o dependency
                z_cuda0 = z_cuda + 1
                z_cuda1 = z_cuda0 @ weight_cuda
                z_cuda2 = 2 * (z_cuda1 + z_cuda)

                # partition 4 on cpu w/o dependency
                y_cpu2 = y_cpu + 5
                y_cpu3 = y_cpu2 @ weight_cpu

                # partition 5 on cuda w/o dependency
                u_cuda0 = z_cuda + 3
                u_cuda1 = u_cuda0 @ weight_cuda
                u_cuda2 = 2 * (u_cuda0 + u_cuda1)

                return x_cuda2, y_cpu1, z_cuda2, y_cpu3, u_cuda2

            x_cuda = torch.randn(3, 3, device="cuda")
            y_cpu = torch.randn(3, 3, device="cpu")
            z_cuda = torch.randn(3, 3, device="cuda")
            weight_cuda = torch.randn(3, 3, device="cuda")
            weight_cpu = torch.randn(3, 3, device="cpu")

            eager_out = f(x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu)

            compiled_f = torch.compile(f, mode="reduce-overhead")
            for _ in range(3):
                compiled_out = compiled_f(
                    x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu
                )
                self.assertEqual(eager_out, compiled_out)

            # the optimal order is
            # [[partition 4 on cpu], [partition 1,3,5 on cuda], [partition 2 on cpu]]
            # since partition2 depends on partition1. So we have 1 cudagraph in total.
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_custom_op_with_no_dependency(self):
            # Two reasons for this:
            # 1. We want to reuse the same mask for many masked_fill calls
            # 2. Prevent inductor from fusing this op into other ops (e.g. masked_fill)
            #    so we can still reorder in scheduler
            @torch.library.custom_op(
                "mylib::create_mask",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def create_mask(
                padded_size: int, original_size: int, device: torch.device
            ) -> torch.Tensor:
                mask = torch.zeros((padded_size,), dtype=torch.bool, device=device)
                mask[original_size:] = True
                return mask

            @create_mask.register_fake
            def _(padded_size, original_size, device):
                return torch.empty((padded_size,), dtype=torch.bool, device=device)

            def f(padded_tensor, original_tensor, weight):
                original_size = original_tensor.size()[0]
                padded_size = padded_tensor.size()[0]

                # element wise op so we don't care padding value
                padded_tensor = padded_tensor + 1
                padded_tensor = torch.nn.functional.relu(padded_tensor)

                # dot product requires padding with 0
                dot_res = padded_tensor.dot(weight)
                padded_tensor += dot_res

                # min requires padding with inf, so we create mask now
                mask = create_mask(padded_size, original_size, padded_tensor.device)
                min_res = torch.min(
                    torch.ops.aten.masked_fill(padded_tensor, mask, float("inf"))
                )

                # max requires padding with inf. we can reuse previous mask
                max_res = torch.max(
                    torch.ops.aten.masked_fill(padded_tensor, mask, -float("inf"))
                )

                return min_res + max_res + padded_tensor

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(padded_size, original_size):
                padded_tensor = torch.randn(padded_size, device="cuda")
                padded_tensor[original_size:] = 0
                original_tensor = torch.randn(original_size, device="meta")

                weight = torch.randn(padded_size, device="cuda")
                eager_out = f(padded_tensor, original_tensor, weight)
                for _ in range(3):
                    compiled_out = compiled_f(padded_tensor, original_tensor, weight)
                    self.assertEqual(eager_out, compiled_out)

            # although custom op `create_mask` happens at the middle of function, reorder
            # moves it to the front so we only have 1 partition. This leads to 1 cudagraph
            run(8, 4)

            # recompilation leads to 1 NEW cudagraph
            run(8, 6)

            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_custom_op_with_no_dependency1(self):
            # wrap with custom op so this is not fused into other ops
            @torch.library.custom_op(
                "mylib::create_size_tensor",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def create_size_tensor(
                tensor: torch.Tensor, device: torch.device
            ) -> torch.Tensor:
                size = tensor.size()[0]
                zero = torch.zeros((), device=device)
                return zero + size

            @create_size_tensor.register_fake
            def _(tensor, device):
                size = tensor.size()[0]
                zero = torch.zeros((), device=device, dtype=torch.int64)
                return zero + size

            def fill(
                padded_tensor: torch.Tensor, original_size: torch.Tensor, value
            ) -> torch.Tensor:
                padded_size = padded_tensor.size()[0]
                size_range = torch.arange(padded_size, device=padded_tensor.device)
                padded_tensor = torch.where(
                    size_range >= original_size, value, padded_tensor
                )
                return padded_tensor

            def f(padded_tensor, original_tensor, weight):
                # element wise op so we don't care padding value
                padded_tensor = padded_tensor + 1
                padded_tensor = torch.nn.functional.relu(padded_tensor)

                # dot product requires padding with 0
                dot_res = padded_tensor.dot(weight)
                padded_tensor += dot_res

                # min requires padding with inf, so we create mask now
                original_size_cuda = create_size_tensor(original_tensor, "cuda")
                padded_tensor = fill(padded_tensor, original_size_cuda, float("inf"))
                min_res = torch.min(padded_tensor)

                # max requires padding with inf. we can reuse previous mask
                padded_tensor = fill(padded_tensor, original_size_cuda, -float("inf"))
                max_res = torch.max(padded_tensor)

                return min_res + max_res + padded_tensor

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(padded_size, original_size):
                padded_tensor = torch.randn(padded_size, device="cuda")
                padded_tensor[original_size:] = 0
                original_tensor = torch.randn(original_size, device="meta")
                weight = torch.randn(padded_size, device="cuda")
                eager_out = f(padded_tensor, original_tensor, weight)
                for _ in range(3):
                    compiled_out = compiled_f(padded_tensor, original_tensor, weight)
                    assert torch.allclose(eager_out, compiled_out)

            # although custom op `create_mask` happens at the middle of function, reorder
            # moves it to the front so we only have 1 partition. This leads to 1 cudagraph
            run(8, 4)

            # recompilation leads to 1 NEW cudagraph
            run(8, 6)

            # reuse previous cudagraph
            run(8, 7)

            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        def test_meta_tensor(self):
            def foobar(x, y):
                return x * 2, y * 3

            foo_c = torch.compile(mode="reduce-overhead")(foobar)
            t = torch.empty((1, 16, 128, 128), device="meta")
            y = torch.rand([64], device="cuda")

            eager_out = foobar(t, y)

            for _ in range(3):
                compiled_out = foo_c(t, y)

            compiled_out = foo_c(t, y)
            self.assertEqual(eager_out, compiled_out)
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        def test_cudagraph_capture_sizes(self):
            torch._inductor.config.triton.cudagraph_capture_sizes = (2, 5, 7)

            def f(x):
                return x + 1

            f = torch.compile(f, mode="reduce-overhead")

            def run(shape):
                x = torch.randn((shape, 5), device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                for _ in range(3):
                    f(x)

            for i in range(1, 10):
                run(i)

            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        def test_cudagraph_capture_sizes1(self):
            torch._inductor.config.triton.cudagraph_capture_sizes = (
                (2, 3),
                (4, 5),
                (6, 2),
                (7, 3),
            )

            def f(x):
                return x + 1

            f = torch.compile(f, mode="reduce-overhead")

            def run(batch_size, seq_len, d):
                x = torch.randn((batch_size, seq_len, d), device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                for _ in range(3):
                    f(x)

            for i in range(2, 10):
                for j in range(2, 10):
                    run(i, j, 8)

            self.assertEqual(self.get_manager().new_graph_id().id, 4)

        def test_cudagraph_capture_sizes2(self):
            torch._inductor.config.triton.cudagraph_capture_sizes = (
                (2, 3, 4),
                (4, 4, 3),
                (3, 4, 4),
                (4, 2, 3),
            )

            def f(x):
                return x + 1

            f = torch.compile(f, mode="reduce-overhead")

            def run(batch_size, seq_len, d):
                x = torch.randn((batch_size, seq_len, d), device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                torch._dynamo.mark_dynamic(x, 2)
                for _ in range(3):
                    f(x)

            for i in range(2, 5):
                for j in range(2, 5):
                    for k in range(2, 5):
                        run(i, j, k)

            self.assertEqual(self.get_manager().new_graph_id().id, 4)

    class TestSAC(TestCase):
        def _make_observer_mode(self):
            class ObserverMode(TorchDispatchMode):
                def __init__(self):
                    super().__init__()
                    self.curr_run = 0
                    self.op_outputs = defaultdict(list)

                def __torch_dispatch__(
                    self,
                    func: OpOverload,
                    types: Sequence[type],
                    args: Sequence[object] = (),
                    kwargs: Mapping[str, object] = immutable_dict(),
                ) -> object:
                    return func(*args, **kwargs)

            return ObserverMode

        def test_simple(self):
            device = "cuda"

            from torch._prims.rng_prims import graphsafe_run_with_rng_state

            ObserverMode = self._make_observer_mode()

            @graphsafe_run_with_rng_state.py_impl(ObserverMode)
            def _(mode, op, *args, **kwargs):
                with no_dispatch():
                    out = graphsafe_run_with_rng_state(op, *args, **kwargs)

                mode.op_outputs[op].append(out)
                return out

            obs = ObserverMode()

            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = torch.randn(4, 4, device=device, requires_grad=True)

            for _ in range(2):
                torch._dynamo.reset()

                def gn(x, y):
                    return torch.sigmoid(torch.rand_like(x) * y) * x

                def fn(x, y):
                    x = torch.sin(x)
                    x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                    x = torch.sin(x)
                    return x

                aot_eager_decomp_partition = functools.partial(
                    aot_eager_decomp_partition_with_mode, mode=obs
                )

                fn = torch.compile(fn, backend=aot_eager_decomp_partition)

                fn(x, y).sum().backward()

            self.assertEqual(len(obs.op_outputs[aten.rand.default]), 4)
            for i in range(2):
                self.assertEqual(
                    obs.op_outputs[aten.rand.default][0 + 2 * i],
                    obs.op_outputs[aten.rand.default][1 + 2 * i],
                )
            self.assertNotEqual(
                obs.op_outputs[aten.rand.default][0],
                obs.op_outputs[aten.rand.default][2],
            )

        def test_cudagraph_uneven_forward_backward(self):
            # torch.compile cudagraphs are difficult to test
            # the rng updating bc is sensitive to duration of pending backwards, etc.
            # this is a short repro to mimic the runtime wrappers integration
            # and show that updating the backward rng state with cudagraphs works:
            def forward():
                state = torch.cuda.get_rng_state()
                perm = torch.randperm(10, device="cuda")
                return state, perm

            def backward(rng_state):
                current_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng_state.cpu())
                perm = torch.randperm(10, device="cuda")
                torch.cuda.set_rng_state(current_state)
                return perm

            def normal_test():
                state, perm = forward()
                repro_perm = backward(state)
                return perm, repro_perm

            def graphsafe_forward():
                perm = torch.randperm(10, device="cuda")
                return perm

            def graphsafe_backward(generator, new_state):
                current_state = generator.graphsafe_get_state()
                generator.graphsafe_set_state(new_state)
                perm = torch.randperm(10, device="cuda")
                generator.graphsafe_set_state(current_state)
                return perm

            def graph_test(generator, capture_cuda_graph):
                if capture_cuda_graph:
                    graph = torch.cuda.CUDAGraph()

                # state should be cloned before the graph
                old_state = generator.graphsafe_get_state()
                new_state = old_state.clone_state()

                if capture_cuda_graph:
                    # state should be register to the graph
                    graph.register_generator_state(new_state)

                    # only capturing the backward
                    with torch.cuda.graph(graph):
                        repro_perm = graphsafe_backward(generator, new_state)

                # some number of uneven forwards
                graphsafe_forward()
                graphsafe_forward()
                graphsafe_forward()

                # state prior to rng invocation
                state = generator.get_state()
                perm = graphsafe_forward()

                new_state.set_state(state)

                if capture_cuda_graph:
                    graph.replay()
                else:
                    repro_perm = graphsafe_backward(generator, new_state)

                return perm, repro_perm

            self.assertEqual(*normal_test())
            generator = torch.cuda.default_generators[0]
            self.assertEqual(*graph_test(generator, capture_cuda_graph=False))
            self.assertEqual(*graph_test(generator, capture_cuda_graph=True))

        def test_cpu_and_cuda_rng(self):
            device = "cuda"

            ObserverMode = self._make_observer_mode()
            from torch._prims.rng_prims import (
                graphsafe_run_with_rng_state,
                run_and_save_rng_state,
                run_with_rng_state,
            )

            for hop in [
                graphsafe_run_with_rng_state,
                run_and_save_rng_state,
                run_with_rng_state,
            ]:

                def make_impl(hop):
                    @hop.py_impl(ObserverMode)
                    def _(mode, *args, **kwargs):
                        with no_dispatch():
                            out = hop(*args, **kwargs)

                        op = None
                        for inp in itertools.chain(args, kwargs.values()):
                            if isinstance(inp, torch._ops.OpOverload):
                                op = inp
                                break
                        assert op is not None
                        if hop is run_and_save_rng_state:
                            mode.op_outputs[op].append(out[1])
                        else:
                            mode.op_outputs[op].append(out)
                        return out

                make_impl(hop)

            obs = ObserverMode()

            def gn(x, y):
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def gn2(x):
                return x * torch.randperm(x.numel(), device=x.device).reshape(x.shape)

            def fn(x, y, z):
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                z = torch.utils.checkpoint.checkpoint(gn2, z, use_reentrant=True)
                return x * z.cuda()

            aot_eager_decomp_partition = functools.partial(
                aot_eager_decomp_partition_with_mode, mode=obs
            )

            fn = torch.compile(fn, backend=aot_eager_decomp_partition)

            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = torch.randn(4, 4, device=device, requires_grad=True)
            z = torch.randn(4, 4, requires_grad=True)

            fn(x, y, z).sum().backward()
            for op in [aten.rand.default, aten.randperm.default]:
                self.assertEqual(len(obs.op_outputs[op]), 2)
                self.assertEqual(
                    obs.op_outputs[op][0],
                    obs.op_outputs[op][1],
                )
                self.assertEqual(
                    obs.op_outputs[op][0].device.type,
                    "cpu" if op == aten.randperm.default else "cuda",
                )

        @parametrize("order", (list(itertools.permutations([0, 1, 2]))))
        def test_uneven_forward_backward(self, order):
            device = "cuda"

            ObserverMode = self._make_observer_mode()
            from torch._prims.rng_prims import graphsafe_run_with_rng_state

            @graphsafe_run_with_rng_state.py_impl(ObserverMode)
            def _(mode, op, *args, **kwargs):
                with no_dispatch():
                    out = graphsafe_run_with_rng_state(op, *args, **kwargs)

                mode.op_outputs[(mode.curr_run, op)].append(out)
                return out

            obs = ObserverMode()

            def gn(x, y):
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def gn2(x):
                return x * torch.randperm(x.numel(), device=x.device).reshape(x.shape)

            def fn(x, y):
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn2, x, use_reentrant=True)
                return x

            aot_eager_decomp_partition = functools.partial(
                aot_eager_decomp_partition_with_mode, mode=obs
            )

            fn_c = torch.compile(fn, backend=aot_eager_decomp_partition)

            torch.manual_seed(0)
            outs = []
            for i in range(len(order)):
                obs.curr_run = i
                x = torch.randn(4, 4, device=device, requires_grad=True)
                y = torch.randn(4, 4, device=device, requires_grad=True)
                outs.append(fn_c(x, y))

            for idx in order:
                obs.curr_run = idx
                outs[idx].sum().backward()

            for run in range(len(order)):
                for op in (aten.rand.default, aten.randperm.default):
                    self.assertEqual(len(obs.op_outputs[(run, op)]), 2)
                    self.assertEqual(
                        obs.op_outputs[(run, op)][0],
                        obs.op_outputs[(run, op)][1],
                    )
                    if run != 0:
                        self.assertNotEqual(
                            obs.op_outputs[(run - 1, op)][0],
                            obs.op_outputs[(run, op)][0],
                        )

        @config.patch(fallback_random=True)
        @config.patch("test_configs.graphsafe_rng_func_ignores_fallback_random", True)
        def _test_cudagraphs_aot_eager_compat_equal(self, device):
            def gn(x, y):
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def fn(x, y):
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                return x

            outs = []
            grads = []

            outs2 = []
            grads2 = []

            compile_fns = [
                lambda fn: torch.compile(fn, backend="aot_eager_decomp_partition"),
                lambda fn: torch.compile(fn, mode="reduce-overhead"),
            ]
            for i, compile_fn in enumerate(compile_fns):
                torch.manual_seed(0)
                for index in range(3):
                    x = torch.randn(4, 4, device=device, requires_grad=True)
                    y = torch.randn(4, 4, device=device, requires_grad=True)

                    out = compile_fn(fn)(x, y)
                    torch.cuda.synchronize()
                    out.sum().backward()
                    if i == 0:
                        outs.append(out.clone())
                        grads.append((x.grad.clone(), y.grad.clone()))
                    else:
                        outs2.append(out.clone())
                        grads2.append((x.grad.clone(), y.grad.clone()))

            self.assertEqual(outs, outs2)
            self.assertEqual(grads, grads2)
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

        def test_cudagraphs_aot_eager_compat_equal(self):
            self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:0"))

        @requires_multigpu()
        def test_cudagraphs_aot_eager_compat_equal_device_one(self):
            self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:1"))

        @config.patch(graph_partition=True)
        def test_graph_partition_cudagraphs_aot_eager_compat_equal(self):
            self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:0"))

        @requires_multigpu()
        def test_multi_device(self):
            def gn(x, y):
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def fn(x, y):
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                return x

            def multi_fn(x, y, a, b):
                return fn(x, y), fn(a, b)

            x = torch.randn(4, 4, device="cuda:0", requires_grad=True)
            y = torch.randn(4, 4, device="cuda:0", requires_grad=True)

            a = torch.randn(4, 4, device="cuda:1", requires_grad=True)
            b = torch.randn(4, 4, device="cuda:1", requires_grad=True)

            # No errors. TODO - get graphs from logging, couldnt figure out how
            multi_fn_c = torch.compile(multi_fn, backend="aot_eager_decomp_partition")

            out = multi_fn_c(x, y, a, b)
            out[0].sum().backward()

        def test_retain_graph(self):
            device = "cuda"

            ObserverMode = self._make_observer_mode()
            from torch._prims.rng_prims import graphsafe_run_with_rng_state

            @graphsafe_run_with_rng_state.py_impl(ObserverMode)
            def _(mode, op, *args, **kwargs):
                with no_dispatch():
                    out = graphsafe_run_with_rng_state(op, *args, **kwargs)

                mode.op_outputs[op].append(out)
                return out

            obs = ObserverMode()

            def gn(x, y):
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def fn(x, y):
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                return x

            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = torch.randn(4, 4, device=device, requires_grad=True)

            aot_eager_decomp_partition = functools.partial(
                aot_eager_decomp_partition_with_mode, mode=obs
            )

            fn = torch.compile(fn, backend=aot_eager_decomp_partition)

            out = fn(x, y).sum()
            out.backward(retain_graph=True)
            out.backward()
            self.assertEqual(len(obs.op_outputs[aten.rand.default]), 3)
            self.assertEqual(
                obs.op_outputs[aten.rand.default][0],
                obs.op_outputs[aten.rand.default][1],
            )
            self.assertEqual(
                obs.op_outputs[aten.rand.default][1],
                obs.op_outputs[aten.rand.default][2],
            )

    instantiate_parametrized_tests(CudaGraphTreeTests)
    instantiate_parametrized_tests(TestSAC)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    if HAS_CUDA:
        run_tests(needs="filelock")
