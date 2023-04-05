# Owner(s): ["module: nvfuser"]

import unittest
import warnings
from functools import partial

import torch
import torch._dynamo as torchdynamo
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.jit_utils import RUN_CUDA

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


def is_networkx_available():
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
@unittest.skipIf(IS_WINDOWS, "TorchDynamo is not supported on Windows")
@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestNvFuserDynamo(TestCase):
    def test_basic(self):
        input1 = make_tensor((2, 4, 8), device="cuda", dtype=torch.float32)
        input2 = make_tensor((2, 4, 8), device="cuda", dtype=torch.float32)

        @torchdynamo.optimize("nvprims_nvfuser")
        def func(a, b):
            return a.sin() + b.cos()

        # No warnings and no errors
        with warnings.catch_warnings(record=True) as w:
            nvfuser_result = func(input1, input2)
            self.assertEqual(len(w), 0)
        eager_result = func.__wrapped__(input1, input2)
        self.assertEqual(eager_result, nvfuser_result)

    @unittest.skipIf(not is_networkx_available(), "networkx not available")
    def test_min_cut(self):
        from functorch.compile import default_partition
        from torch._dynamo.backends.nvfuser import nvprims_fw_bw_partition_fn

        def get_fw_bw_graph(f, inps, partitioner):
            from functorch.compile import aot_function

            # Helper functions are taken from functorch/test_aotdispatch.py
            def extract_graph(fx_g, _, graph_cell):
                graph_cell[0] = fx_g
                return fx_g

            fw_graph_cell = [None]
            bw_graph_cell = [None]
            aot_function(
                f,
                fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
                bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
                partition_fn=partitioner,
            )(*inps).sum().backward()
            return (fw_graph_cell[0], bw_graph_cell[0])

        def get_ins_outs(fx_g):
            ins = []
            outs = []
            for n in fx_g.graph.nodes:
                if n.op == "placeholder":
                    ins.append(n)
                elif n.op == "output":
                    outs = tuple(n.args[0])
            return ins, outs

        def get_num_ins_outs(fx_g):
            return tuple(len(i) for i in get_ins_outs(fx_g))

        def func(x):
            return x * x * x

        input1 = make_tensor(
            (3,), device="cpu", dtype=torch.float32, requires_grad=True
        )
        fw_graph, bw_graph = get_fw_bw_graph(func, [input1], default_partition)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))

        input1 = make_tensor(
            (3,), device="cpu", dtype=torch.float32, requires_grad=True
        )
        fw_graph, bw_graph = get_fw_bw_graph(func, [input1], nvprims_fw_bw_partition_fn)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

    def test_batch_norm_implicit_dtype_promotion(self):
        input1 = make_tensor((2, 3, 4, 5), device="cuda", dtype=torch.float32)
        input2 = make_tensor((5, 5), device="cuda", dtype=torch.float32)
        w = make_tensor((3), device="cuda", dtype=torch.float32)
        b = make_tensor((3), device="cuda", dtype=torch.float32)

        @torchdynamo.optimize("nvprims_nvfuser")
        def func(mat1, mat2, w, b):
            o = torch.matmul(mat1, mat2)
            return torch.batch_norm(o, w, b, None, None, True, 1e-2, 1e-5, True)

        # No warnings and no errors
        with torch.cuda.amp.autocast():
            with warnings.catch_warnings(record=True) as warning:
                nvfuser_result = func(input1, input2, w, b)
                self.assertEqual(len(warning), 0)
            eager_result = func.__wrapped__(input1, input2, w, b)
            self.assertEqual(eager_result, nvfuser_result)

    def test_dtype_correctness(self):
        input1 = make_tensor((2, 4, 8), device="cuda", dtype=torch.float16)

        @torchdynamo.optimize("nvprims_nvfuser")
        def func(a):
            tmp = a + 1.0
            # nvfuser would promote output to fp32 in math, FusionDefinition should cast output dtype back
            return torch.where(tmp > 0, tmp, 0.0)

        # No warnings and no errors
        with warnings.catch_warnings(record=True) as w:
            nvfuser_result = func(input1)
            self.assertEqual(len(w), 0)
        eager_result = func.__wrapped__(input1)
        self.assertEqual(eager_result, nvfuser_result)


if __name__ == "__main__":
    run_tests()
