#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "ATen/ATen.h"

#include "torch/csrc/utils/disallow_copy.h"

#include "torch/csrc/jit/fusers/fuser_interface.h"
#include "torch/csrc/jit/fusers/cuda/annotated_graph.h"
#include "torch/csrc/jit/fusers/cuda/tensor_desc.h"
#include "torch/csrc/jit/fusers/cuda/concat_desc.h"
#include "torch/csrc/jit/fusers/cuda/cuda_check.h"

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "nvrtc.h"

#include <string>
#include <vector>

namespace torch { namespace jit { namespace cudafuser {

struct CUDAFusionFunction : public CompiledFusionFunction {
  TH_DISALLOW_COPY_AND_ASSIGN(CUDAFusionFunction);

  friend class CUDAFuser;
  
  CUDAFusionFunction(const std::string& name, AnnotatedGraph& agraph);

  // Note: Creates new tensors for outputs
  virtual void launch(
    at::ArrayRef<at::Tensor> inputs
  , std::vector<at::Tensor>& outputs) override;

  // Note: expects outputs to be pre-allocated
  void launch_with_tensors(
    at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

  const std::vector<TensorDesc>& outputDescriptors() const {
    return output_desc;
  }

  // Note: this call to cuModuleUnload is intentionally unchecked
  // Shutdown unloads the driver and can destroy the fusion functions, and
  // if the driver is unloaded first then cuModuleUnload will fail. 
  // This is OK.
  virtual ~CUDAFusionFunction() override {
    cuModuleUnload(module);
  }

private:
  uint64_t get_rand_offset(uint32_t numel);

  // arguments is a list of pointers to the arguments for the compiled CUDA/CPU
  // code.
  // The format of arguments is suitable for directly passing to a call to
  // cuLaunchKernel as the kernel arguments.
  // Currently the first argument is a pointer to numel (for passing to
  // CUDA code), and the remainder are pointers to the TensorInfo<T> structs
  // that compiled code uses to load Tensor data.
  // launch_with_tensors handles packing at::Tensors into this arguments array.
  // CPU code uses the same convension so that launch_with_tensors can be shared.
  void launch_raw(uint32_t numel, void** arguments);

  std::string name;
  bool has_random;

  // for debugging
  std::string compilation_unit;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;

  // same size as output_desc, describes whether
  // an output is actually a concatenation of
  // many subtensors that the fusion group produces
  std::vector<ConcatDesc> concat_desc;

  std::vector<char> ptx;
  CUmodule module;
  CUfunction function;

  // we record prop/device so if they are availiable for launch heuristics
  // querying at launch is too slow for device properties.
  int device;
  cudaDeviceProp prop;
  int blockSize = 128;
  int maxBlocks;
};

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
