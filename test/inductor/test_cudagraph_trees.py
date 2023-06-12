# Owner(s): ["module: inductor"]
import contextlib
import functools
import gc
import importlib
import sys
import unittest
import warnings

import torch

import torch._dynamo
import torch.nn as nn
from torch._inductor import config
from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl
from torch.testing import FileCheck

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    TEST_CUDA_GRAPH,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase as TorchTestCase,
)

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

HAS_MULTIGPU = HAS_CUDA and torch.cuda.device_count() >= 2
aten = torch.ops.aten
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")
requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, "requires multiple cuda devices"
)


def cdata(t):
    return t.untyped_storage()._cdata


class TestCase(TorchTestCase):
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
                (self.device_idx if not device_index else device_index)
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

            foo_opt = torch._dynamo.optimize()(foo)
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

        def test_function_compiled_multiple_times(self):
            def foo(x):
                y = foo2(x)
                y2 = foo2(y)
                return y + y2

            def foo2(x):
                torch._dynamo.graph_break()
                return x * x * x

            foo_opt = torch._dynamo.optimize()(foo)
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

            @torch._dynamo.optimize()
            def foo2(x):
                return x + 4

            foo_opt = torch._dynamo.optimize()(foo)

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

            foo_opt = torch._dynamo.optimize()(foo)
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

        def test_forward_backward_not_called(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x, y):
                x_out = x * x * x
                torch._dynamo.graph_break()
                y_out = y * y * y
                return x_out, y_out

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

        def test_accumulate_multiple_recordings(self):
            def foo(x):
                y = x + x + x
                torch._dynamo.graph_break()
                if y.sum() <= 0:
                    return y
                else:
                    return y * 10

            foo_opt = torch._dynamo.optimize()(foo)

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

            foo_opt = torch._dynamo.optimize()(foo)

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
                return (x + x + x), torch.zeros([0], device="cuda")

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
                    FileCheck().check("at::cuda::getNewWorkspace").check(
                        "at::cuda::blas::gemm<float>"
                    ).run(str(e))

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
                @torch._dynamo.optimize()
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
            @torch._dynamo.optimize()
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

            foo = torch._dynamo.optimize()(foo_unopt)

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

        def test_forward_generation(self):
            def foo(x):
                return x * x * x

            def foo2(x):
                return x * 12

            foo_opt = torch._dynamo.optimize()(foo)
            foo2_opt = torch._dynamo.optimize()(foo2)
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

            torch._inductor.cudagraph_mark_step_begin()
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            torch._inductor.cudagraph_mark_step_begin()
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))
            self.assertFalse(self.get_manager().new_graph_id().id == 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
