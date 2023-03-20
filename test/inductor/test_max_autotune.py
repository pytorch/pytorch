# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import do_bench
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch import multiprocessing as mp
import unittest
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm
from torch._inductor.ir import Buffer, FixedLayout, InputBuffer, TensorBox
from torch._inductor.select_algorithm import AlgorithmSelectorCache, ChoiceCaller
from torch._inductor.virtualized import V
from torch._inductor.graph import GraphLowering
from torch._inductor.config import override_configs
from torch.fx.experimental.proxy_tensor import make_fx

torch.set_float32_matmul_precision("high")

def benchmark_choice(choice, args, out, expected_out, timings):
    result = choice.benchmark(*args, out=out)
    if expected_out is not None:
        torch.testing.assert_close(out, expected_out) 

    timings.copy_(torch.tensor(result))

class FailChoiceCaller(ChoiceCaller):
    def benchmark(self, *args, out):
        raise RuntimeError("This choice caller will always throw")

class TestDoBench(TestCase):
    def _create_buffer(self, name, shape):
        return Buffer(
            name,
            FixedLayout(torch.device("cuda:0"), torch.float32, shape)
        )

    @unittest.skip("Pickle StorageBox fail with https://gist.github.com/171e5ab404b7855dee2dfa1d9f093442")
    def test_pickle_storage_box(self):
        from pickle import loads, dumps
        buf = TensorBox.create(self._create_buffer("buf0", (2, 3))).data
        loads(dumps(buf))

    @unittest.skip("Pickle fx.Node fail with https://gist.github.com/9c289e895d7091d7ec787c67bc3c0d70")
    def test_pickle_fx_node(self):
        from pickle import loads, dumps
        gm = make_fx(lambda: torch.zeros(2, 3))()
        nd = next(iter(gm.graph.nodes))
        loads(dumps(nd))

    def test_benchmark_choice_in_subproc(self):
        gm = make_fx(lambda: torch.zeros(2, 3))()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is neede to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))
    
            layout = FixedLayout(torch.device("cuda:0"), torch.float32, (2, 2))
    
            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)
    
            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            # expected_out = (mat1 @ mat2) + (mat3 @ mat4)
            expected_out = None
    
            choice = aten_mm_plus_mm.bind((buf1, buf2, buf3, buf4), layout)
            # use a tensor since the mutation to a python list in a sub process
            # is not synced back to the parent process
            timings = torch.zeros(3, dtype=torch.float32)
            child = mp.Process(target=benchmark_choice, args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings))
            child.start()
            child.join()
            self.assertEqual(0, child.exitcode)
            print(f"timings is {timings}, out {out}, expected_out {expected_out}")

    def test_benchmark_choice_fail_in_subproc(self):
        gm = make_fx(lambda: torch.zeros(2, 3))()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is neede to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))
    
            layout = FixedLayout(torch.device("cuda:0"), torch.float32, (2, 2))
    
            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)
    
            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            expected_out = (mat1 @ mat2) + (mat3 @ mat4)
    
            choice = FailChoiceCaller("fail_choice_caller", [], None)

            # use a tensor since python list is not synced back
            timings = torch.zeros(3, dtype=torch.float32)
            child = mp.Process(target=benchmark_choice, args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings))
            child.start()
            child.join()
            self.assertNotEqual(0, child.exitcode)

    def test_max_autotune_mm_plus_mm(self):
        """
        This crash previously due to a triton issue: https://github.com/openai/triton/issues/1298 .
        With autotuning in subprocess, we don't crash anymore.
        """
        m, n, k = 2048, 1536, 64

        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        a = torch.randn(m, k).cuda()
        b = torch.randn(k, n).cuda()
        c = torch.randn(m, k).cuda()
        d = torch.randn(k, n).cuda()

        with override_configs(max_autotune=True, autotune_in_subproc=True):
            torch.compile(mm_plus_mm)(a, b, c, d)

if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
