#pragma once

#include <ATen/core/ivalue.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <cuda.h>

#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace executor_utils {

// Include all the functions we might need in generated code
std::string kernelPreamble();

void validateKernelInputs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const c10::Device& device);

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device);

StatefulExpressionEvaluator statefulBindInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    Fusion* fusion,
    GpuLower* lower = nullptr);

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = CUfunction();
};

NvrtcFunction nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id);

} // namespace executor_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
