#pragma once
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <c10/core/DeviceType.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Should this actually be in launch params?
struct TORCH_CUDA_CU_API CompileOptions {
  c10::Device device = c10::Device(c10::DeviceType::CUDA, 0);
};

class TORCH_CUDA_CU_API FusionExecutor : public NonCopyable {
 public:
  // Unsafe compilation that's useful for debugging kernels, iterating over
  // slight modifications of a generated kernel
  void debugCompileFusionFromStr(
      Fusion* fusion,
      const std::string& code,
      const std::string& name,
      int id,
      CompileOptions options = CompileOptions());

  void compileFusion(Fusion* fusion, CompileOptions options = CompileOptions());

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<IValue>& inputs,
      const std::vector<at::Tensor>& outputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      const c10::optional<size_t>& opt_code = c10::nullopt);

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<IValue>& inputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      const c10::optional<size_t>& opt_code = c10::nullopt) {
    return runFusion(inputs, {}, launch_constraints, opt_code);
  }

  // function to query whether a `FusionExecutor` has a compiled kernel to
  // execute
  bool compiled() const {
    return fusion_id_ != -1;
  };

  void evictCache(size_t cache_id) {
    executor_entry_lookup_.erase(cache_id);
  }

  // TODO: strides would also be important when we handle permutations in
  //       codegen.
  // struct used to hold necessary information to launch compiled kernel on a
  // given input set.
  struct ExecutorEntry {
    bool init = false;
    LaunchParams launch_params;
    std::vector<std::vector<int64_t>> output_sizes;
    std::vector<at::ScalarType> output_types;
    std::vector<std::vector<int64_t>> empty_buffer_sizes;
    std::vector<at::ScalarType> empty_buffer_types;
    std::vector<std::vector<int64_t>> zero_buffer_sizes;
    std::vector<at::ScalarType> zero_buffer_types;
    uint64_t rand_offset;
  };

  Kernel* kernel() const {
    return lowered_.kernel();
  }

 private:
  struct GlobalBuffers {
    std::vector<at::Tensor> empty_buffers;
    std::vector<at::Tensor> zero_buffers;
  };

  std::string kernelName() const {
    std::stringstream ss;
    ss << "kernel" << fusion_id_;
    return ss.str();
  }

  static std::string kernelNamespace() {
    return "CudaCodeGen";
  }

  // Add preamble and wrap in namespace
  std::string getStructuredCode(const std::string& kernel);

  LaunchParams computeLaunchParams(
      const LaunchParams& launch_constraints,
      StatefulExpressionEvaluator& see);

  uint64_t computeSharedMemory(
      StatefulExpressionEvaluator& see,
      const std::vector<kir::Allocate*>& buffers,
      bool align_padding = false,
      uint64_t total = 0);

  // return a pair of vector of tensors, where tensors in the first vector are
  // not initialized, while the second vector contains zero-initiliazed tensors
  GlobalBuffers allocGlobalVals(StatefulExpressionEvaluator& see);

  std::vector<at::Tensor> allocOutputs(StatefulExpressionEvaluator& see);

  void setUsedTVs();

  const std::vector<TensorView*>& getUsedTVs() const {
    return used_tvs_;
  };

 private:
  Fusion fusion_;

  // TODO(kir): caching the values here is no longer needed
  bool has_block_reductions = false;
  bool has_grid_reductions = false;
  bool has_block_broadcasts = false;

  CompileOptions options_;
  size_t max_device_smem = std::numeric_limits<size_t>().max();
  executor_utils::NvrtcFunction compiled_kernel_;

  // TensorViews actually used in the kernel.
  std::vector<TensorView*> used_tvs_;

  // Counter to be used for kernel name.
  int fusion_id_ = -1;
  static int fusion_id_counter_;

  GpuLower lowered_;

  // lookup table to take short cut to retrieve recorded information in order to
  // launch kernels without re-inference parameters.
  std::unordered_map<size_t, ExecutorEntry> executor_entry_lookup_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
