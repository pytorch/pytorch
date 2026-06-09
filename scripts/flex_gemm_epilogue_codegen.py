#!/usr/bin/env python3
"""Flex GEMM epilogue fusion codegen experiment from PR #181963.

Example:
    python scripts/flex_gemm_epilogue_codegen.py --backend QUACK --epilogue relu
    python scripts/flex_gemm_epilogue_codegen.py --backend QUACK --gemm-op addmm
"""

import enum
import re
from pathlib import Path
from typing import Annotated

import torch
import torch._inductor.config as inductor_config
import torch.utils._pytree as pytree
import typer
from torch._higher_order_ops import (
    gemm_epilogue_fusion,
    grouped_mm_epilogue,
    matmul_epilogue,
)
from torch._inductor.utils import run_and_get_code
from transformer_nuggets.utils.benchmark import profiler


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class GemmOp(str, enum.Enum):
    MM = "mm"
    ADDMM = "addmm"
    BMM = "bmm"
    BADDBMM = "baddbmm"
    MATMUL_2D = "matmul-2d"
    MATMUL_3D = "matmul-3d"
    GROUPED_MM_2D_2D = "grouped-mm-2d-2d"
    GROUPED_MM_2D_3D = "grouped-mm-2d-3d"


EPILOGUES = {
    "identity": lambda acc: acc,
    "relu": lambda acc: acc.relu(),
    "affine-relu": lambda acc: (acc * 0.5 + 1.0).relu(),
    "gelu": lambda acc: torch.nn.functional.gelu(acc),
    "silu": lambda acc: torch.nn.functional.silu(acc),
    "sigmoid": lambda acc: torch.sigmoid(acc),
    "tanh": lambda acc: torch.tanh(acc),
    "exp": lambda acc: torch.exp(acc),
    "abs": lambda acc: torch.abs(acc),
    "square": lambda acc: acc * acc,
    "clamp": lambda acc: acc.clamp(min=0.0, max=6.0),
    "leaky-relu": lambda acc: torch.nn.functional.leaky_relu(acc, 0.01),
    "where-leaky-relu": lambda acc: torch.where(acc > 0, acc, acc * 0.01),
    "sqrt-abs": lambda acc: torch.sqrt(torch.abs(acc) + 1.0),
    "rsqrt-abs": lambda acc: torch.rsqrt(torch.abs(acc) + 1.0),
    "normalize-sum-n": lambda acc: acc / acc.sum(dim=1, keepdim=True),
    "relu-local-sum-n32": lambda acc: (
        acc.relu(),
        acc.float().view(acc.shape[0], -1, 32).sum(-1),
    ),
}
AUX_EPILOGUES = {
    # QUACK currently supports one captured aux tensor for plain mm.
    "mul-aux": ("tile", lambda acc, aux: acc * aux),
    "add-aux-relu": ("tile", lambda acc, aux: (acc + aux).relu()),
    "mul-row-aux": ("row", lambda acc, aux: acc * aux),
    "add-row-aux-relu": ("row", lambda acc, aux: (acc + aux).relu()),
    "mul-col-aux": ("col", lambda acc, aux: acc * aux),
    "add-col-aux-relu": ("col", lambda acc, aux: (acc + aux).relu()),
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_problem(
    gemm_op: GemmOp,
    m: int,
    n: int,
    k: int,
    batch: int,
    groups: int,
    dtype: torch.dtype,
    epilogue: str,
    backend: str,
    alpha: float,
    beta: float,
):
    kernel_options = {"backend": backend}
    epilogue_fn = EPILOGUES.get(epilogue)
    aux_epilogue = AUX_EPILOGUES.get(epilogue)
    if gemm_op == GemmOp.MM:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        if aux_epilogue is not None:
            aux_kind, aux_epilogue_fn = aux_epilogue
            aux_shape = {"tile": (m, n), "row": (1, n), "col": (m, 1)}[aux_kind]
            aux = torch.randn(*aux_shape, device="cuda", dtype=dtype)

            def fn(x, y, aux_tensor):
                return gemm_epilogue_fusion(
                    torch.ops.aten.mm.default,
                    (x, y),
                    lambda acc: aux_epilogue_fn(acc, aux_tensor),
                    kernel_options=kernel_options,
                )

            return fn, (a, b, aux), f"({m}, {k}) x ({k}, {n}) with {aux_kind} aux {aux_shape}"

        assert epilogue_fn is not None

        def fn(x, y):
            return gemm_epilogue_fusion(
                torch.ops.aten.mm.default,
                (x, y),
                epilogue_fn,
                kernel_options=kernel_options,
            )

        return fn, (a, b), f"({m}, {k}) x ({k}, {n})"

    if aux_epilogue is not None:
        raise NotImplementedError(
            f"epilogue {epilogue!r} uses an aux tensor and is currently supported only with --gemm-op mm"
        )
    assert epilogue_fn is not None

    if gemm_op == GemmOp.ADDMM:
        bias = torch.randn(m, n, device="cuda", dtype=dtype)
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)

        def fn(input, mat1, mat2):
            return gemm_epilogue_fusion(
                torch.ops.aten.addmm.default,
                (input, mat1, mat2),
                epilogue_fn,
                gemm_kwargs={"alpha": alpha, "beta": beta},
                kernel_options=kernel_options,
            )

        return fn, (bias, a, b), f"({m}, {n}) + ({m}, {k}) x ({k}, {n})"

    if gemm_op == GemmOp.BMM:
        a = torch.randn(batch, m, k, device="cuda", dtype=dtype)
        b = torch.randn(batch, k, n, device="cuda", dtype=dtype)

        def fn(x, y):
            return gemm_epilogue_fusion(
                torch.ops.aten.bmm.default,
                (x, y),
                epilogue_fn,
                kernel_options=kernel_options,
            )

        return fn, (a, b), f"{batch} x ({m}, {k}) x ({k}, {n})"

    if gemm_op == GemmOp.BADDBMM:
        bias = torch.randn(batch, m, n, device="cuda", dtype=dtype)
        a = torch.randn(batch, m, k, device="cuda", dtype=dtype)
        b = torch.randn(batch, k, n, device="cuda", dtype=dtype)

        def fn(input, batch1, batch2):
            return gemm_epilogue_fusion(
                torch.ops.aten.baddbmm.default,
                (input, batch1, batch2),
                epilogue_fn,
                gemm_kwargs={"alpha": alpha, "beta": beta},
                kernel_options=kernel_options,
            )

        return fn, (bias, a, b), f"{batch} x ({m}, {n}) + ({m}, {k}) x ({k}, {n})"

    if gemm_op == GemmOp.MATMUL_2D:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)

        def fn(x, y):
            return matmul_epilogue(x, y, epilogue_fn, kernel_options=kernel_options)

        return fn, (a, b), f"matmul ({m}, {k}) x ({k}, {n})"

    if gemm_op == GemmOp.MATMUL_3D:
        a = torch.randn(batch, m, k, device="cuda", dtype=dtype)
        b = torch.randn(batch, k, n, device="cuda", dtype=dtype)

        def fn(x, y):
            return matmul_epilogue(x, y, epilogue_fn, kernel_options=kernel_options)

        return fn, (a, b), f"matmul {batch} x ({m}, {k}) x ({k}, {n})"

    if gemm_op == GemmOp.GROUPED_MM_2D_2D:
        a = torch.randn(m, k * groups, device="cuda", dtype=dtype)
        b = torch.randn(n, k * groups, device="cuda", dtype=dtype).t()
        # QUACK grouped-mm 2d/2d currently requires m-major/contiguous inputs.
        a = a.t().contiguous().t()
        b = b.contiguous()
        offs = torch.arange(k, k * groups + 1, k, device="cuda", dtype=torch.int32)

        def fn(x, y, group_offsets):
            return grouped_mm_epilogue(
                x,
                y,
                epilogue_fn,
                offs=group_offsets,
                kernel_options=kernel_options,
            )

        return fn, (a, b, offs), f"grouped 2d/2d groups={groups}, ({m}, {k}) x ({k}, {n})"

    if gemm_op == GemmOp.GROUPED_MM_2D_3D:
        a = torch.randn(m * groups, k, device="cuda", dtype=dtype)
        b = torch.randn(groups, n, k, device="cuda", dtype=dtype).transpose(-2, -1)
        offs = torch.arange(m, m * groups + 1, m, device="cuda", dtype=torch.int32)

        def fn(x, y, group_offsets):
            return grouped_mm_epilogue(
                x,
                y,
                epilogue_fn,
                offs=group_offsets,
                kernel_options=kernel_options,
            )

        return fn, (a, b, offs), f"grouped 2d/3d groups={groups}, ({m}, {k}) x ({k}, {n})"

    raise AssertionError(f"unsupported gemm_op: {gemm_op}")


def main(
    out_dir: Annotated[
        Path,
        typer.Option(help="Directory for exported graph and generated code artifacts."),
    ] = Path(__file__).resolve().parents[1] / "agent_space" / "gemm_epilogue_codegen",
    m: Annotated[int, typer.Option(help="Rows of A/output.")] = 2048,
    n: Annotated[int, typer.Option(help="Columns of B/output.")] = 2048,
    k: Annotated[int, typer.Option(help="Reduction dimension.")] = 2048,
    dtype: Annotated[
        str,
        typer.Option(help="Input dtype.", case_sensitive=False),
    ] = "float16",
    gemm_op: Annotated[
        GemmOp,
        typer.Option(help="GEMM op to try."),
    ] = GemmOp.MM,
    batch: Annotated[int, typer.Option(help="Batch size for bmm/baddbmm/matmul-3d.")] = 8,
    groups: Annotated[int, typer.Option(help="Number of groups for grouped-mm variants.")] = 2,
    alpha: Annotated[float, typer.Option(help="alpha kwarg for addmm/baddbmm.")] = 0.5,
    beta: Annotated[float, typer.Option(help="beta kwarg for addmm/baddbmm.")] = 1.25,
    epilogue: Annotated[
        str,
        typer.Option(help="Epilogue to try; see EPILOGUES/AUX_EPILOGUES in this script."),
    ] = "relu",
    backend: Annotated[
        str,
        typer.Option(help="Requested GEMM epilogue backend: TRITON, CUTEDSL, or QUACK."),
    ] = "QUACK",
    profile: Annotated[bool, typer.Option(help="Whether to run the compiled code with a profiler.")] = False,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment expects CUDA to be available")
    if dtype not in DTYPES:
        raise typer.BadParameter(f"dtype must be one of {', '.join(DTYPES)}")
    all_epilogues = EPILOGUES.keys() | AUX_EPILOGUES.keys()
    if epilogue not in all_epilogues:
        raise typer.BadParameter(
            f"epilogue must be one of {', '.join(sorted(all_epilogues))}"
        )

    torch.manual_seed(0)
    fn, args, shape_desc = make_problem(
        gemm_op,
        m,
        n,
        k,
        batch,
        groups,
        DTYPES[dtype],
        epilogue,
        backend,
        alpha,
        beta,
    )

    out_dir = out_dir / gemm_op.value
    out_dir.mkdir(parents=True, exist_ok=True)
    eager = fn(*args)

    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    with inductor_config.patch({"max_autotune_gemm": False}):
        actual, codes = run_and_get_code(compiled, *args)
    torch.testing.assert_close(actual, eager, atol=1e-2, rtol=1e-2)
    actual_leaves = pytree.tree_leaves(actual)
    eager_leaves = pytree.tree_leaves(eager)
    max_abs_diff = max(
        (actual_leaf - eager_leaf).abs().max().item()
        for actual_leaf, eager_leaf in zip(actual_leaves, eager_leaves)
    )

    kernels = []
    for i, code in enumerate(codes):
        write_text(out_dir / f"inductor_output_{i}.py", code)
        kernels.extend(
            re.findall(
                r"async_compile\.triton\([^\n]+, '''\n(.*?)\n'''",
                code,
                re.DOTALL,
            )
        )

    for i, kernel in enumerate(kernels):
        write_text(out_dir / f"triton_kernel_{i}.py", kernel)

    summary = [
        "FlexGEMM epilogue fusion",
        f"torch: {torch.__file__}",
        f"git_version: {torch.version.git_version}",
        f"gemm_op: {gemm_op.value}",
        f"shape: {shape_desc}",
        f"dtype: {dtype}",
        f"epilogue: {epilogue}",
        f"backend: {backend}",
        f"alpha: {alpha}",
        f"beta: {beta}",
        f"inductor_outputs: {len(codes)}",
    ]
    summary.extend(f"  - {out_dir / f'inductor_output_{i}.py'}" for i in range(len(codes)))
    summary.append(f"triton_kernels: {len(kernels)}")
    summary.extend(f"  - {out_dir / f'triton_kernel_{i}.py'}" for i in range(len(kernels)))
    summary.append(f"max_abs_diff: {max_abs_diff}")

    write_text(out_dir / "summary.txt", "\n".join(summary) + "\n")
    print("\n".join(summary))

    if profile:
        print("\nProfiling the compiled code...")
        with profiler(f"gemm_epilogue_{gemm_op.value}_{epilogue}_backend_{backend}"):
            compiled(*args)


if __name__ == "__main__":
    typer.run(main)
