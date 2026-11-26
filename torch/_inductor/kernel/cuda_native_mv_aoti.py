# mypy: allow-untyped-defs

from pathlib import Path

import triton.language as tl

import torch

from .. import ir
from ..runtime.cuda_native_mv import _copy_kernel, gemv, gemv_mma, zero
from ..virtualized import V


def codegen_cuda_native_mv(kernel, wrapper):
    """Package generated GEMV kernels and dispatch from runtime C++ geometry."""
    if not wrapper._cuda_native_mv_helpers_written:
        for header in (
            "ATen/cuda/CUDABlas.h",
            "ATen/Functions.h",
            "c10/cuda/CUDAGuard.h",
            "torch/csrc/inductor/aoti_torch/utils.h",
            "regex",
        ):
            wrapper.include_extra_header(header)
        wrapper.header.splice(
            (Path(__file__).parent / "templates/cuda_native_mv_runtime.cpp").read_text()
        )
        wrapper._cuda_native_mv_helpers_written = True

    x, vector = kernel.inputs
    device = x.get_device()
    dtype = x.get_dtype()
    complex_input = dtype.is_complex
    components = 2 if complex_input else 1
    physical_dtype = (
        (torch.float64 if dtype == torch.complex128 else torch.float32)
        if complex_input
        else dtype
    )
    acc_dtype = (
        torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    )
    ctx = kernel.get_name() + "_gemv"
    stream = wrapper.write_get_raw_stream(device.index, "")
    matrix_name, vector_name = kernel.codegen_args()
    wrapper.writeline("{")
    wrapper.writeline(
        f"CudaNativeMvContext {ctx}({matrix_name}, {vector_name}, "
        f"reinterpret_cast<cudaStream_t>({stream}));"
    )
    symbols = {}
    for key in (
        "M",
        "K",
        "S0",
        "S1",
        "VS",
        "CTAS",
        "ACTIVE_THREADS",
        "INNER_CHUNK",
        "INNER_STRIDE",
        "SWAP",
        "PRODUCT",
        "PRE_SCALE",
        "K_SLICE",
    ):
        symbol = V.graph.sizevars.shape_env.create_unbacked_symint().node.expr
        symbols[key] = symbol
        wrapper.writeline(f"const int64_t {symbol} = {ctx}.{key};")

    def buffer(field, scalar_type):
        name = ctx + "_" + field
        wrapper.writeline(f"auto& {name} = {ctx}.{field};")
        result = ir.Buffer(name=name, layout=ir.FixedLayout(device, scalar_type, [1]))
        V.graph.name_to_buffer[name] = result
        V.graph.unaligned_buffers.add(name)
        return result

    matrix = buffer("matrix", physical_dtype)
    vec = buffer("vector", physical_dtype)
    source_matrix = buffer("source_matrix", physical_dtype)
    source_vector = buffer("source_vector", physical_dtype)
    partial = buffer("partial", acc_dtype)
    output = ir.Buffer(
        name=kernel.get_name(),
        layout=ir.FixedLayout(device, physical_dtype, kernel.get_size()),
    )

    def scalar(expression):
        symbol = V.graph.sizevars.shape_env.create_unbacked_symint().node.expr
        wrapper.writeline(f"const int64_t {symbol} = {expression};")
        return symbol

    def launch(fn, grid, **kwargs):
        kwargs.update(
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            enable_precise_libdevice=True,
        )
        name, triton_meta, inductor_meta, extras = (
            wrapper.define_user_defined_triton_kernel(
                fn,
                [],
                kwargs,
                [],
                [],
                [grid],
                None,
                (
                    "num_warps",
                    "num_stages",
                    "enable_fp_fusion",
                    "enable_precise_libdevice",
                ),
            )
        )
        # Runtime geometry cannot supply fake inputs for execution-based autotuning.
        if name not in wrapper.kernel_autotune_names:
            wrapper.kernel_autotune_calls.writeline(
                f"{name}.use_first_config(stream=0)"
            )
            wrapper.kernel_autotune_names.add(name)
        keys, raw, args, types = [], [], [], []
        for parameter in fn.params:
            if parameter.is_constexpr:
                continue
            value = kwargs[parameter.name]
            keys.append(parameter.name)
            raw.append(value)
            args.append(
                value.codegen_reference() if isinstance(value, ir.IRNode) else value
            )
            types.append(
                value.get_dtype() if isinstance(value, ir.IRNode) else type(value)
            )
        for value in extras:
            keys.append("")
            raw.append(value)
            args.append(value)
            types.append(type(value))
        wrapper.generate_kernel_call(
            name,
            args,
            arg_types=types,
            raw_args=raw,
            raw_keys=keys,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            device=device,
            triton=True,
        )

    wrapper.writeline(f"if ({ctx}.M != 0) {{")
    wrapper.writeline(f"if ({ctx}.K == 0) {{")
    count = scalar(f"{ctx}.M * {components}")
    launch(zero, [(count + 255) // 256, 1, 1], OUT=output, N=count)
    wrapper.writeline("} else {")
    for field, source, dest, is_matrix in (
        ("matrix", source_matrix, matrix, True),
        ("vector", source_vector, vec, False),
    ):
        wrapper.writeline(f"if ({ctx}.copy_{field}) {{")
        count = scalar(f"{ctx}.source_{field}.numel() * {components}")
        copy_args = dict(
            SOURCE=source, DEST=dest, N=count, K=symbols["K"], COMPONENTS=components
        )
        for key, expression in (
            ("S0", f"{ctx}.source_{field}.stride(0)" if is_matrix else "0"),
            ("S1", f"{ctx}.source_{field}.stride(-1)"),
            ("D0", f"{ctx}.{field}.stride(0)" if is_matrix else "0"),
            ("D1", f"{ctx}.{field}.stride(-1)"),
            ("NEGATIVE", f"{ctx}.source_{field}.is_neg()"),
            ("CONJUGATE", f"{ctx}.source_{field}.is_conj()"),
        ):
            copy_args[key] = scalar(expression)
        launch(_copy_kernel, [(count + 255) // 256, 1, 1], **copy_args)
        wrapper.writeline("}")

    options = dict(
        X=matrix,
        V=vec,
        P=partial,
        O=output,
        **symbols,
        ACC_TYPE=tl.float64 if acc_dtype == torch.float64 else tl.float32,
        COMPLEX=complex_input,
    )
    m, ctas = symbols["M"], symbols["CTAS"]
    wrapper.writeline(f"bool {ctx}_dispatched = false;")
    if dtype in (torch.float16, torch.bfloat16):
        wrapper.writeline(f"if ({ctx}.KIND == 4 || {ctx}.KIND == 5) {{")
        mma = {
            key: options[key]
            for key in ("X", "V", "M", "K", "S0", "S1", "VS", "CTAS", "K_SLICE")
        }
        wrapper.writeline(f"if ({ctx}.KIND == 4) {{")
        launch(gemv_mma, [(m + 15) // 16, ctas, 1], P=output, **mma)
        wrapper.writeline("} else {")
        launch(gemv_mma, [(m + 15) // 16, ctas, 1], P=partial, **mma)
        launch(gemv, [m, 1, 1], **options, THREADS=1, KIND=3, FINAL=True)
        wrapper.writeline("}")
        wrapper.writeline(f"{ctx}_dispatched = true;")
        wrapper.writeline("}")
    for kind in (0, 2, 3):
        for threads in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            if kind == 2 and threads < 32:
                continue
            wrapper.writeline(
                f"if ({ctx}.KIND == {kind} && {ctx}.THREADS == {threads}) {{"
            )
            launch(
                gemv, [m, ctas, 1], **options, THREADS=threads, KIND=kind, FINAL=False
            )
            launch(gemv, [m, 1, 1], **options, THREADS=threads, KIND=kind, FINAL=True)
            wrapper.writeline(f"{ctx}_dispatched = true;")
            wrapper.writeline("}")
    wrapper.writeline(
        f'AOTI_TORCH_CHECK({ctx}_dispatched, "Unpackaged native GEMV geometry");'
    )
    wrapper.writeline("}")
    wrapper.writeline("}")
    wrapper.writeline("}")
