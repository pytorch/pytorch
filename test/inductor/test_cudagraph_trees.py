# Owner(s): ["module: inductor"]
import contextlib
import functools
import gc
import importlib
import sys
import unittest
import warnings
from unittest import mock

import torch
import torch._dynamo.config as dynamo_config
import torch.nn as nn
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl
from torch._inductor.cudagraph_utils import FunctionID
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    parametrize,
    skipIfRocm,
    TEST_CUDA_GRAPH,
    TEST_WITH_ASAN,
)
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

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


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


if HAS_CUDA and not TEST_WITH_ASAN:

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
                def __init__(self):
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
                def __init__(self):
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

                def complex_memory_overlap_new(t):
                    return True

                try:
                    prev = torch._inductor.compile_fx.complex_memory_overlap
                    torch._inductor.compile_fx.complex_memory_overlap = (
                        complex_memory_overlap_new
                    )
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)
                finally:
                    torch._inductor.compile_fx.complex_memory_overlap = prev

            # we should not have cudagraph'd the backwards
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        def test_cache_hit_forward_miss_backward(self):
            # Test that we don't cache cudagraphs, skipping cudagraphs on backward on a cache miss

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            def complex_memory_overlap_new(t):
                return True

            # Run forwards, fx graph should cache miss
            for _ in range(3):
                torch._dynamo.reset()
                counters.clear()
                FxGraphCache.clear()

                with mock.patch(
                    "torch._inductor.compile_fx.complex_memory_overlap",
                    new=complex_memory_overlap_new,
                ):
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

            # we should not have cudagraph'd anything
            assert self.get_manager() is None

        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
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
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

            # we should not have cudagraph'd anything
            assert self.get_manager() is None

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
                def __init__(self):
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
                def __init__(self):
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

            x = torch.rand([10, 10], device="cuda", requires_grad=True)
            param_c = cdata(m.weight)
            for _ in range(3):
                out1, alias_1, alias_2 = foo(m, x)
                self.assertEqual(len({param_c, cdata(alias_1), cdata(alias_2)}), 1)

                out2 = foo2(out1)
                out2.sum().backward()
                self.assertEqual(cdata(out1), cdata(out2))

            node = self.curr_node()
            first_node = next(node._path_from_root)
            self.assertFalse(first_node.unaliased_in_all_paths[0])
            self.assertTrue(first_node.cached_tensor_outputs[0] is None)

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
        @unittest.skipIf(not IS_LINUX, "cpp contexts are linux only")
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
                    self.assertTrue(
                        "at::cuda::blas::gemm<float>" in str(e)
                        or "at::cuda::blas::gemm_internal_cublas<float>" in str(e)
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
                def __init__(self):
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

            with self.assertRaisesRegex(Exception, "overwritten by a subsequent run."):
                out + out

            foo(inp)

            with self.assertRaisesRegex(Exception, "overwritten by a subsequent run."):
                out2 + out2

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
                "skipping cudagraphs due to cpu device (_local_scalar_dense)"
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

            FileCheck().check("skipping cudagraphs due to ['incompatible ops']").run(
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

        def test_storage_access_error(self):
            x = torch.rand([4], device="cuda")
            torch._C._set_storage_access_error_msg(x, "custom error msg")

            with self.assertRaisesRegex(Exception, "custom error msg"):
                device = x.untyped_storage()

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
                r"static input data pointer changed.\n"
                r"input name: primals_2. data pointer changed from .* to .*. input stack trace:(?s).*"
                r"input name: primals_3. data pointer changed from .* to .*. input stack trace:.*,"
                r" in forward\n.* self.static_tensor.add\_\(torch.ones\(\(2, 2\), device=\"cuda\"\)\).*\n",
            ):
                self.curr_node().run(
                    [foo.goo.linear.weight, foo.goo.linear.bias, foo.static_tensor, inp]
                )

        def _run_iter(self, param, fn):
            fwd_output = fn(torch.ones(2, 2), param)
            fwd_output.sum().backward()
            grad_output = param.grad.clone().detach()
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
                    grad_output = mod.weight.grad.clone().detach()
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
                    self.register_buffer("buf", buf)

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

                    for _ in range(5):
                        # Change static tensor address
                        fn_compiled.param.data = torch.rand([2, 2], device="cuda")
                        fn_compiled(torch.rand([2, 2], device="cuda")).sum().backward()

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
            ).run(
                captured_output[0]
            )
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
                def __init__(self):
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
                def __init__(self):
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

    instantiate_parametrized_tests(CudaGraphTreeTests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
