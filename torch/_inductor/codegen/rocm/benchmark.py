import logging
import sys

import pandas  # type: ignore[import-untyped]

import torch
from torch._dynamo import config as dynconfig
from torch._inductor import config
from torch._inductor.codegen.rocm.ck_template import CKTemplate

log = logging.getLogger(__name__)


def generate_inputs(M, N, K, tensor_options, layout, f=torch.randn):
    if layout[0] == "r":
        a = f(M, K, **tensor_options)
    elif layout[0] == "c":
        a = f(K, M, **tensor_options).transpose(0, 1)
    else:
        a = None

    if layout[1] == "r":
        b = f(K, N, **tensor_options)
    elif layout[1] == "c":
        b = f(N, K, **tensor_options).transpose(0, 1)
    else:
        b = None

    if layout[2] == "r":
        out = torch.empty(M, N, **tensor_options)
    elif layout[2] == "c":
        out = torch.empty(N, M, **tensor_options).transpose(0, 1)
    else:
        out = None
    return a, b, out


def main(gemm_shape_csv, layout, dtype):
    df = pandas.read_csv(gemm_shape_csv)

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    tensor_options = {"device": "cuda", "dtype": dtype}

    problem_instances = df[["M", "K", "N"]].values

    def mm(a, b, out):
        return torch.mm(a, b, out=out)

    for M, K, N in problem_instances:
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CK,Triton,ATen",
                "compile_threads": 64,
            }
        ), dynconfig.patch(
            {"cache_size_limit": len(problem_instances) + 1}
        ), torch.no_grad():
            a, b, out = generate_inputs(M, N, K, tensor_options, layout)
            Y_compiled = torch.compile(mm, dynamic=False)(a, b, out)
            Y = mm(a, b, out)
            try:
                torch.testing.assert_close(Y_compiled, Y)
            except AssertionError as e:
                log.error(e)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].lower() == "torchbench":
        gemm_shape_csv = "https://raw.githubusercontent.com/pytorch/benchmark/main/torchbenchmark/operators/gemm/amd.csv"
    else:
        gemm_shape_csv = sys.argv[1]

    if len(sys.argv) < 3:
        layout = "rcr"
    else:
        layout = sys.argv[2].lower()

    if len(sys.argv) < 4:
        dtype = torch.half
    else:
        # as long as the mapping is 1:1 this is fine
        ck_dtype_to_torch = {v: k for k, v in CKTemplate._TORCH_DTYPE_TO_CK.items()}
        dtype = ck_dtype_to_torch[sys.argv[3].upper()]
    main(gemm_shape_csv, layout, dtype)
