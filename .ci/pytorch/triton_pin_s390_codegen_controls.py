"""Native CPU codegen controls against the captured original layout failure."""

import itertools
import json
import os
from pathlib import Path
import platform
import subprocess

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]


def kernel_source(label, explicit_fma=False):
    source = (ROOT / "aten/src/ATen/native/cpu/GridSamplerKernel.cpp").read_text()
    source = source.replace("REGISTER_DISPATCH(grid_sampler_2d_cpu_kernel, &grid_sampler_2d_cpu_kernel_impl)", "")
    source = source.replace("REGISTER_DISPATCH(grid_sampler_2d_backward_cpu_kernel, &grid_sampler_2d_backward_cpu_kernel_impl)", "")
    source = source.replace("#define TORCH_ASSERT_NO_OPERATORS", "#include <ATen/Functions.h>\n#include <torch/library.h>")
    if explicit_fma:
        replacements = {
            "(in + Vec(static_cast<scalar_t>(1))) * Vec(scaling_factor) - Vec(static_cast<scalar_t>(0.5))":
                "fmadd(in + Vec(static_cast<scalar_t>(1)), Vec(scaling_factor), Vec(static_cast<scalar_t>(-0.5)))",
            "((A * x - Vec(5) * A) * x + Vec(8) * A) * x - Vec(4) * A":
                "fmadd(fmadd(fmadd(A, x, -Vec(5) * A), x, Vec(8) * A), x, -Vec(4) * A)",
            "((A + Vec(2)) * x - (A + Vec(3))) * x * x + Vec(1)":
                "fmadd(fmadd(A + Vec(2), x, -(A + Vec(3))) * x, x, Vec(1))",
            "(Vec(-3) * A * x - Vec(10) * A ) * x - Vec(8) * A":
                "fmadd(fmadd(Vec(-3) * A, x, -Vec(10) * A), x, -Vec(8) * A)",
            "(Vec(-3) * (A + Vec(2)) * x - Vec(2) * (A + Vec(3))) * x":
                "fmadd(Vec(-3) * (A + Vec(2)), x, -Vec(2) * (A + Vec(3))) * x",
            "(Vec(3) * (A + Vec(2)) * x - Vec(2) * (A + Vec(3))) * x":
                "fmadd(Vec(3) * (A + Vec(2)), x, -Vec(2) * (A + Vec(3))) * x",
            "(Vec(3) * A * x - Vec(10) * A) * x + Vec(8) * A":
                "fmadd(fmadd(Vec(3) * A, x, -Vec(10) * A), x, Vec(8) * A)",
            "gx = gx - val * gOut * coeff_x_grad[i] * coeff_y[j]":
                "gx = fmadd(-val * gOut * coeff_x_grad[i], coeff_y[j], gx)",
            "gy = gy - val * gOut * coeff_y_grad[j] * coeff_x[i]":
                "gy = fmadd(-val * gOut * coeff_y_grad[j], coeff_x[i], gy)",
        }
        for old, new in replacements.items():
            assert old in source, old
            source = source.replace(old, new)
    # Keep the complete original dispatch/specialization compilation context.
    # An indirect call mirrors DispatchStub and prevents constant propagation
    # from reducing this to the smaller bicubic-only probe.
    source += r'''
namespace at::native {
std::vector<Tensor> pin_backward_LABEL(const Tensor& input, const Tensor& grid,
                                      const Tensor& grad_output_) {
  TORCH_CHECK(input.scalar_type() == kFloat && grid.scalar_type() == kFloat);
  auto grad_input = at::zeros(input.sizes(), input.options());
  auto grad_grid = at::empty(grid.sizes(), grid.options());
  decltype(&grid_sampler_2d_backward_cpu_kernel_impl) volatile fn =
      &grid_sampler_2d_backward_cpu_kernel_impl;
  fn(grad_input, grad_grid, grad_output_, input, grid, 2, 0, false, {true, true});
  return {grad_input, grad_grid};
}
}
TORCH_LIBRARY(triton_pin_LABEL, m) {
  m.def("bicubic_backward(Tensor input, Tensor grid, Tensor grad_output) -> Tensor[]",
        &at::native::pin_backward_LABEL);
}
'''.replace("LABEL", label)
    return source


def investigate(data, report, debug):
    native = platform.machine() == "s390x"
    os.environ["MAX_JOBS"] = "2"
    if native:
        os.environ["CXX"] = "/opt/rh/gcc-toolset-14/root/usr/bin/c++"
    inp, grid = data["primals"]
    cotangent = data["cotangents"]
    ni, ng = data["noncontig_primals"]
    nc = data["noncontig_cotangents"]
    assert data["sample_args"][1:] == (2, 0, False)
    layouts = list(itertools.product([inp, ni], [grid, ng], [cotangent, nc]))
    truth = torch.ops.aten.grid_sampler_2d_backward(cotangent.double(), inp.double(), grid.double(),
                                                   2, 0, False, [True, True])
    baseline = [torch.ops.aten.grid_sampler_2d_backward(cv, iv, gv, 2, 0, False, [True, True])
                for iv, gv, cv in layouts]
    flags = ["-O3", "-std=c++20", "-fno-math-errno", "-fno-trapping-math"]
    flags += (["-march=z15", "-mvx", "-mzvector", "-DCPU_CAPABILITY=ZVECTOR", "-DCPU_CAPABILITY_ZVECTOR"]
              if native else ["-mavx2", "-mfma", "-mf16c", "-DCPU_CAPABILITY=AVX2", "-DCPU_CAPABILITY_AVX2"])
    results = {}
    for label, explicit, extra in [("original", False, []),
                                   ("contraction_off", False, ["-ffp-contract=off"]),
                                   ("explicit_fma", True, [])]:
        folder = debug / label
        folder.mkdir(parents=True, exist_ok=True)
        cpp = folder / "kernel.cpp"
        cpp.write_text(kernel_source(label, explicit))
        load(name="triton_pin_" + label, sources=[str(cpp)], extra_cflags=flags + extra,
             build_directory=str(folder), is_python_module=False, verbose=True)
        op = getattr(torch.ops, "triton_pin_" + label).bicubic_backward
        outputs = [op(iv, gv, cv) for iv, gv, cv in layouts]
        rows = []
        for output, original in zip(outputs, baseline):
            rows.append({"max_difference_to_wheel": [(a-b).abs().max().item() for a,b in zip(output, original)],
                         "max_layout_difference": [(a-b).abs().max().item() for a,b in zip(output, outputs[0])],
                         "max_error_to_float64": [(a.double()-b).abs().max().item() for a,b in zip(output, truth)],
                         "grid_at_original_failure": output[1][1,1,4,1].item()})
        results[label] = rows
        print("S390_NATIVE_CODEGEN_POINT", label, json.dumps(rows), flush=True)
        if native:
            with (folder / "assembly.txt").open("w") as assembly:
                subprocess.run(["objdump", "-d", "-C", str(folder / ("triton_pin_"+label+".so"))],
                               stdout=assembly, check=True)
        (report / "codegen-results.json").write_text(json.dumps(results, indent=2))
    print("S390_NATIVE_CODEGEN_CONTROLS", json.dumps(results), flush=True)


if __name__ == "__main__":
    import triton_pin_s390_controls as controls
    investigate(torch.load(controls.DEBUG / "original-layout-repro.pt", weights_only=False),
                controls.REPORT, controls.DEBUG)
