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

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
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
            self.prev_enabled = config.triton.cudagraphs
            self.tapes_enabled = config.triton.cudagraph_trees
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            self.device_idx = torch.rand([0], device="cuda").device.index
            warnings.filterwarnings("ignore")

        def tearDown(self):
            super().tearDown()
            torch._dynamo.reset()
            gc.collect()
            config.triton.cudagraphs = self.prev_enabled
            config.triton.cudagraph_trees = self.tapes_enabled
            self.assertIsNone(self.get_manager())
            self.assertEqual(all_live_block_count(), 0)
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

        def cudagraphify_impl(self, *args, **kwargs):
            return tree_cudagraphify_impl(
                *args,
                **kwargs,
                device_index=self.device_idx,
                is_backward=False,
                is_inference=True,
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

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_dies_between_checkpoint(self):
            def foo(args):
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            foo_cg = self.cudagraphify_impl(foo, [inp], ())
            foo_cg([inp])
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

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_no_longer_in_pool(self):
            def foo(args):
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            foo_cg = self.cudagraphify_impl(foo, [inp], ())
            x1, x2 = foo_cg([inp])

            def foo2(args):
                x = args[0]
                args.clear()
                return [x * x * x]

            foo2_cg = self.cudagraphify_impl(foo2, [x1], ())
            foo2_cg([x1])

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
            foo_cg = self.cudagraphify_impl(foo, [inp], ())
            foo_cg([inp])
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
            print("Input ID", id(inp))
            out = foo(inp)
            out.sum().backward()

            self.assertEqual(self.get_root_children(), [1])

            # the three saved tensors should die in the backward
            # we kept alive the output
            self.assertEqual(self.curr_node().expected_dead_indices_before_graph, [])
            self.assertEqual(
                self.curr_node().expected_dead_indices_after_graph,
                [(0, 1), (0, 2), (0, 3)],
            )

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
                    return x + 3

                inp = torch.rand([20, 20], device="cuda:1")

                foo_cg = tree_cudagraphify_impl(
                    foo, [inp], (), device_index=1, is_backward=False, is_inference=True
                )
                self.assertEqual(foo_cg([inp]), foo([inp]))

                self.assertTrue(self.get_manager(device_index=0) is None)
                self.assertFalse(self.get_manager(device_index=1) is None)

            test()
            self.assertTrue(self.get_manager(device_index=1) is None)

        def test_warnings_on_dealloc(self):
            @torch.compile()
            def foo(x):
                return x * x * x

            inp = torch.rand([4], device="cuda")
            out = foo(inp)
            warnings.resetwarnings()
            with warnings.catch_warnings(record=True) as w:
                foo(inp)

            self.assertTrue(len(w) == 1)
            self.assertTrue("x * x * x" in str(w[0]))

        def test_single_stream_use(self):
            @torch.compile()
            def foo(x):
                return (x * x * x).relu()

            inp = torch.rand([4], device="cuda", requires_grad=True)
            streams = set()

            for _ in range(3):
                foo(inp).sum().backward()

            streams = {seg["stream"] for seg in get_all_cudagraph_segments()}
            self.assertEqual(len(streams), 1)

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

            self.assertEqual(self.get_manager().forwards_with_pending_backwards, 2)

            out2.sum().backward()

            self.assertEqual(self.get_manager().forwards_with_pending_backwards, 0)

            del out
            del out2

            out = foo_opt(ones.detach())
            self.assertEqual(self.get_manager().forwards_with_pending_backwards, 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
