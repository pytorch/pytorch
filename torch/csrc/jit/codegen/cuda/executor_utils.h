#pragma once

#include <ATen/core/ivalue.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <ATen/cuda/CUDAContext.h>

#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
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

// TODO(kir): rewrite in terms of Kernel inputs
void validateKernelInputs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const c10::Device& device);

// TODO(kir): rewrite in terms of Kernel outputs
void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device);

// Returns if vectorizing the aten value by word size is possible
bool canVectorize(const IValue& aten_val, int word_size);

// Returns if vectorizing the aten value by word size is possible
bool canVectorize(
    TensorView* fusion_tv,
    int word_size,
    GpuLower& lower,
    kir::ExpressionEvaluator& expr_eval);

// TODO(kir): rewrite in terms of Kernel tensors
void validateVectorizedTensors(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    GpuLower& lower,
    kir::ExpressionEvaluator& expr_eval);

//! Bind kernel input values to runtime values
kir::ExpressionEvaluator bindKernelInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    kir::Kernel* kernel);

//! Bind fusion input values to runtime values
TORCH_CUDA_CU_API ExpressionEvaluator
bindFusionInputs(const at::ArrayRef<IValue>& aten_inputs, Fusion* fusion);

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = CUfunction();
};

NvrtcFunction nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id,
    c10::optional<int> opt_block_size = c10::nullopt);

} // namespace executor_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
