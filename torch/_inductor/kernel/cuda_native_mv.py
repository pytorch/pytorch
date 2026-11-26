# mypy: allow-untyped-defs

import sympy

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from .. import ir, lowering as L
from ..runtime.cuda_native_mv import _native_options
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..virtualized import V
from .mm_common import load_kernel_template


@SymbolicGridFn
def cuda_native_mv_grid(*args):
    meta = args[-1]
    return (
        (meta["M"] + 15) // 16 if meta["KIND"] in (4, 5) else meta["M"],
        1 if meta["FINAL"] else meta["CTAS"],
        1,
    )


@SymbolicGridFn
def cuda_native_mv_aligned_grid(*args):
    meta = args[-1]
    return ((meta["M"] + 15) // 16, 1 if meta["FINAL"] else meta["MAX_CTAS"], 1)


cuda_native_mv_aligned_template = TritonTemplate(
    name="cuda_native_mv_aligned",
    grid=cuda_native_mv_aligned_grid,
    source=load_kernel_template("triton_cuda_native_mv_aligned"),
)


cuda_native_mv_template = TritonTemplate(
    name="cuda_native_mv",
    grid=cuda_native_mv_grid,
    source=load_kernel_template("triton_cuda_native_mv"),
)


cuda_native_complex_mv_template = TritonTemplate(
    name="cuda_native_complex_mv",
    grid=cuda_native_mv_grid,
    source=load_kernel_template("triton_cuda_native_complex_mv"),
)


cuda_native_mv_mma_template = TritonTemplate(
    name="cuda_native_mv_mma",
    grid=cuda_native_mv_grid,
    source=load_kernel_template("triton_cuda_native_mv_mma"),
)


def _copy_complex(x, preserve_format):
    real, imag = L.complex_components(x)
    parts = L.cat((L.unsqueeze(real, -1), L.unsqueeze(imag, -1)), -1)
    if preserve_format:
        sizes = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
        strides = [V.graph.sizevars.guard_int(s) for s in x.get_stride()]
        cloned = torch.empty_strided(sizes, strides, device="meta").clone()
        parts = ir.ExternKernel.require_exact_strides(
            parts, [2 * s for s in cloned.stride()] + [1]
        )
    else:
        parts = ir.ExternKernel.require_contiguous(parts)
    return ir.TensorBox.create(
        ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
    )


def cuda_native_mv(x, vector):
    r"""Generate GEMV kernels using native CUDA launch geometry and arithmetic."""
    node = V.graph.current_node
    if node is not None:
        for index, buffer in enumerate((x, vector)):
            argument = node.args[index]
            if isinstance(argument, torch.fx.Node):
                constrained = L.constrain_to_fake_tensor(
                    buffer, argument.meta.get("val")
                )
                if index == 0:
                    x = constrained
                else:
                    vector = constrained
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    complex_dtype = x.get_dtype() if x.get_dtype().is_complex else None
    x.realize()
    vector.realize()
    geometry = (*x.get_size(), *x.get_stride(), *vector.get_stride())
    if any(not isinstance(value, (int, sympy.Integer)) for value in geometry):
        V.graph.preserve_input_alignment(
            (x.get_name(), vector.get_name()),
            "runtime CUDA mv requires original input pointer alignment",
        )
        return ir.TensorBox.create(
            RuntimeCudaNativeMv(
                ir.FixedLayout(device, x.get_dtype(), [x.get_size()[0]]),
                (x, vector),
                python_kernel_name="cuda_native_mv_runtime",
            )
        )
    if V.graph.is_negative_buffer(x.get_name()) or V.graph.is_conjugate_buffer(
        x.get_name()
    ):
        x = _copy_complex(x, True) if complex_dtype is not None else L.clone(x)
    if V.graph.is_negative_buffer(vector.get_name()) or V.graph.is_conjugate_buffer(
        vector.get_name()
    ):
        vector = (
            _copy_complex(vector, True)
            if complex_dtype is not None
            else L.clone(vector)
        )
    m, k = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    if m == 0 or k == 0:
        if complex_dtype is not None:
            parts = L.full(
                [m, 2],
                0,
                device=x.get_device(),
                dtype=torch.float64
                if complex_dtype == torch.complex128
                else torch.float32,
            )
            return ir.TensorBox.create(
                ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
            )
        return L.full([m], 0, device=x.get_device(), dtype=x.get_dtype())
    s0, s1 = [V.graph.sizevars.guard_int(s) for s in x.get_stride()]
    if s0 == 1 and s1 >= max(1, m):
        transposed, lda = False, s1
    elif s1 == 1 and s0 >= max(1, k):
        transposed, lda = True, s0
    else:
        if complex_dtype is not None:
            x = _copy_complex(x, False)
        else:
            x = ir.ExternKernel.require_contiguous(x)
        s0, s1 = k, 1
        transposed, lda = True, k
    vs = V.graph.sizevars.guard_int(vector.get_stride()[0])
    if vs == 0:
        vector = (
            _copy_complex(vector, False)
            if complex_dtype is not None
            else ir.ExternKernel.require_contiguous(vector)
        )
        vs = 1
    vs = max(1, vs)
    V.graph.preserve_input_alignment(
        (x.get_name(), vector.get_name()),
        "strict CUDA mv requires original input pointer alignment",
    )
    element_size = torch.empty((), dtype=x.get_dtype()).element_size()
    alignments = [
        (15, 0),
        *((2 * value - 1, value) for value in (8, 4, 2) if value >= element_size),
    ]
    plans = {}
    with unset_fake_temporarily(), torch.cuda.device(x.get_device()):
        for matrix_mask, matrix_value in alignments:
            for vector_mask, vector_value in alignments:
                plan = torch._C._cuda_getGemvLaunchPlan(
                    x.get_dtype(),
                    k if transposed else m,
                    m if transposed else k,
                    lda,
                    vs,
                    transposed,
                    matrix_value,
                    vector_value,
                )
                options = _native_options(plan, x.get_dtype(), m, k, s0, s1, vs)
                key = tuple(options.items())
                if key not in plans:
                    plans[key] = (options, [])
                plans[key][1].append(
                    (matrix_mask, matrix_value, vector_mask, vector_value)
                )
    if len(plans) > 1:
        return _aligned_mv(x, vector, tuple(plans.values()))
    options = next(iter(plans.values()))[0]
    ctas, kind = options["CTAS"], options["KIND"]
    acc_dtype = (
        torch.float64
        if x.get_dtype() in (torch.float64, torch.complex128)
        else torch.float32
    )
    if kind in (4, 5):
        first = cuda_native_mv_mma_template.generate(
            input_nodes=(x, vector),
            layout=ir.FixedLayout(
                device,
                acc_dtype if kind == 5 else x.get_dtype(),
                [m, ctas] if kind == 5 else [m],
            ),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=kind == 4,
            **options,
        ).output_node()
        if kind == 4:
            return first
        final_options = options.copy()
        final_options["KIND"] = 3
        return cuda_native_mv_template.generate(
            input_nodes=(x, vector, first),
            layout=ir.FixedLayout(device, x.get_dtype(), [m]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=True,
            **final_options,
        ).output_node()
    if complex_dtype is not None:
        x = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, x)
        )
        vector = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, vector)
        )
        if kind in (2, 3):
            partial = cuda_native_complex_mv_template.generate(
                input_nodes=(x, vector),
                layout=ir.FixedLayout(device, acc_dtype, [m, ctas, 2]),
                num_stages=1,
                num_warps=4,
                triton_meta={"enable_fp_fusion": False},
                FINAL=False,
                **options,
            ).output_node()
            inputs = (x, vector, partial)
        else:
            inputs = (x, vector)
        parts = cuda_native_complex_mv_template.generate(
            input_nodes=inputs,
            layout=ir.FixedLayout(device, acc_dtype, [m, 2]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=True,
            **options,
        ).output_node()
        return ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_complex.default, parts)
        )
    if kind in (2, 3):
        partial = cuda_native_mv_template.generate(
            input_nodes=(x, vector),
            layout=ir.FixedLayout(device, acc_dtype, [m, ctas]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=False,
            **options,
        ).output_node()
        inputs = (x, vector, partial)
    else:
        inputs = (x, vector)
    return cuda_native_mv_template.generate(
        input_nodes=inputs,
        layout=ir.FixedLayout(device, x.get_dtype(), [m]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    ).output_node()


def _aligned_mv(x, vector, plans):
    options = plans[0][0]
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    complex_dtype = x.get_dtype() if x.get_dtype().is_complex else None
    acc_dtype = (
        torch.float64
        if x.get_dtype() in (torch.float64, torch.complex128)
        else torch.float32
    )
    max_ctas = max(p[0]["CTAS"] for p in plans)
    keys = (
        "KIND",
        "CTAS",
        "THREADS",
        "ACTIVE_THREADS",
        "INNER_CHUNK",
        "INNER_STRIDE",
        "SWAP",
        "PRODUCT",
        "PRE_SCALE",
    )
    metadata = dict(
        M=options["M"],
        K=options["K"],
        S0=options["S0"],
        S1=options["S1"],
        VS=options["VS"],
        MAX_CTAS=max_ctas,
        ACC_TYPE=options["ACC_TYPE"],
        COMPLEX=complex_dtype is not None,
        PLANS=tuple(
            (tuple(conditions), *(p[key] for key in keys), p.get("K_SLICE", 0))
            for p, conditions in plans
        ),
    )
    if complex_dtype is not None:
        x = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, x)
        )
        vector = ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_real.default, vector)
        )
    partial_shape = [options["M"], max_ctas] + (
        [2] if complex_dtype is not None else []
    )
    partial = cuda_native_mv_aligned_template.generate(
        input_nodes=(x, vector),
        layout=ir.FixedLayout(device, acc_dtype, partial_shape),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=False,
        **metadata,
    ).output_node()
    V.graph.no_fuse_buffer_names.add(partial.get_name())
    output_shape = [options["M"]] + ([2] if complex_dtype is not None else [])
    output = cuda_native_mv_aligned_template.generate(
        input_nodes=(x, vector, partial),
        layout=ir.FixedLayout(
            device,
            acc_dtype if complex_dtype is not None else x.get_dtype(),
            output_shape,
        ),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **metadata,
    ).output_node()
    if complex_dtype is not None:
        return ir.TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view_as_complex.default, output)
        )
    return output


class RuntimeCudaNativeMv(ir.ExternKernelOut):
    def codegen(self, wrapper):
        if V.graph.aot_mode:
            from .cuda_native_mv_aoti import codegen_cuda_native_mv

            codegen_cuda_native_mv(self, wrapper)
        elif V.graph.cpp_wrapper:
            wrapper.include_extra_header(
                "torch/csrc/inductor/cpp_wrapper/lazy_triton_compile.h"
            )
            matrix, vector = self.codegen_args()
            output = self.codegen_reference()
            lines = f"""
RAIIPyObject module(PyImport_ImportModule("torch._inductor.runtime.cuda_native_mv"));
AOTI_TORCH_CHECK(module, "Failed to import the runtime GEMV launcher");
RAIIPyObject launcher(PyObject_GetAttrString(module, "cuda_native_mv"));
AOTI_TORCH_CHECK(launcher, "Failed to load the runtime GEMV launcher");
RAIIPyObject matrix(convertArgToPython({matrix}));
RAIIPyObject vector(convertArgToPython({vector}));
RAIIPyObject output(convertArgToPython({output}));
AOTI_TORCH_CHECK(matrix && vector && output, "Failed to wrap GEMV arguments");
RAIIPyObject args(PyTuple_Pack(2, matrix.get(), vector.get()));
RAIIPyObject kwargs(PyDict_New());
AOTI_TORCH_CHECK(args && kwargs, "Failed to construct GEMV arguments");
AOTI_TORCH_CHECK(PyDict_SetItemString(kwargs, "out", output.get()) == 0,
                 "Failed to construct the GEMV output argument");
RAIIPyObject result(PyObject_Call(launcher, args, kwargs));
AOTI_TORCH_CHECK(result, "Failed to launch generated GEMV kernels");
"""
            wrapper.writelines(wrapper.generate_scoped_gil_acquire([], lines))
        else:
            wrapper.add_import_once(
                "from torch._inductor.runtime.cuda_native_mv import cuda_native_mv as cuda_native_mv_runtime"
            )
            super().codegen(wrapper)
