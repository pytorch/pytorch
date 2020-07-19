#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace executor_utils {

// Include all the functions we might need in generated code
std::string kernelPreamble();

bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    c10::Device device,
    std::stringstream& msg);

bool validateKernelArgScalar(
    const c10::TypePtr& arg_type,
    const Val* param,
    std::stringstream& msg);

bool validateKernelArg(
    const c10::IValue& arg,
    const Val* param,
    c10::Device device,
    std::stringstream& msg);

void validateKernelInputs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    c10::Device device);

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    c10::Device device);

struct NvrtcFunction {
 public:
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
