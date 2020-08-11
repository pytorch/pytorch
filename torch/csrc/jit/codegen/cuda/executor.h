#pragma once
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <c10/core/DeviceType.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Should this actually be in launch params?
struct TORCH_CUDA_API CompileOptions {
  c10::Device device = c10::Device(c10::DeviceType::CUDA, 0);
};

class TORCH_CUDA_API FusionExecutor {
 public:
  void compileFusion(Fusion* fusion, CompileOptions options = CompileOptions());

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<IValue>& inputs,
      const std::vector<at::Tensor>& outputs,
      const LaunchParams& launch_constraints = LaunchParams());

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<IValue>& inputs,
      const LaunchParams& launch_constraints = LaunchParams()) {
    return runFusion(inputs, {}, launch_constraints);
  }

  // function to query whether a `FusionExecutor` has a compiled kernel to
  // execute
  bool compiled() const {
    return compiled_;
  };

 private:
  std::string kernelName() const {
    std::stringstream ss;
    ss << "kernel" << fusion_id;
    return ss.str();
  }

  static std::string kernelNamespace() {
    return "CudaCodeGen";
  }

  // Add preamble and wrap in namespace
  std::string getStructuredCode(const std::string& kernel);

  LaunchParams computeLaunchParams(
      const at::ArrayRef<IValue>& aten_inputs,
      const LaunchParams& launch_constraints,
      EvaluationContext& ec);

  std::vector<at::Tensor> allocGlobalVals(EvaluationContext& ec);

  std::vector<at::Tensor> allocOutputs(EvaluationContext& ec);

 private:
  bool compiled_ = false;

  Fusion fusion_;

  CompileOptions options_;

  executor_utils::NvrtcFunction compiled_kernel;

  // State of the fusion that's important
  bool has_random = false;

  // Counter to be used for kernel name.
  int fusion_id = -1;
  static int fusion_id_counter;

  GPULower lowered;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
