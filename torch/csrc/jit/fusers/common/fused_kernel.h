#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER
#pragma once

#include "torch/csrc/jit/fusers/common/annotated_graph.h"
#include "torch/csrc/jit/fusers/common/tensor_desc.h"
#include "torch/csrc/jit/fusers/common/partition_desc.h"

#include "torch/csrc/utils/disallow_copy.h"

#include "ATen/ATen.h"

#include <string>
#include <cstdint>
#include <vector>

namespace torch { namespace jit {

std::tuple<std::vector<PartitionDesc>, std::vector<PartitionDesc>, bool> emitCompilationUnit(
  std::ostream& out
, const std::string& name
, AnnotatedGraph& agraph
, bool use_cuda);

struct FusedKernel {
  TH_DISALLOW_COPY_AND_ASSIGN(FusedKernel);

  FusedKernel(
    const std::string& name
  , AnnotatedGraph& agraph)
  : name{name}
  , input_desc{agraph.input_desc}
  , output_desc{agraph.output_desc} { }

  virtual ~FusedKernel() = default;

  // expects outputs to be pre-allocated
  void launch_with_tensors(
    at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

  // creates new tensors for outputs
  void launch(
    at::ArrayRef<at::Tensor> inputs
  , std::vector<at::Tensor>& outputs);
  
  const std::vector<TensorDesc>& outputDescriptors() const {
    return output_desc;
  }

protected:

  virtual at::Backend backend() const = 0;

  // arguments is a list of pointers to the arguments for the compiled CUDA/CPU
  // code.
  // The format of arguments is suitable for directly passing to a call to
  // cuLaunchKernel as the kernel arguments.
  // Currently the first argument is a pointer to numel (for passing to
  // CUDA code), and the remainder are pointers to the TensorInfo<T> structs
  // that compiled code uses to load Tensor data.
  // launch_with_tensors handles packing at::Tensors into this arguments array.
  // CPU code uses the same convension so that launch_with_tensors can be shared.
  virtual void launch_raw(uint32_t numel, void** arguments) = 0;

  virtual uint64_t get_rand_offset(uint32_t numel) = 0;
  bool has_random;
  std::string name;
  // We keep these around for debugging
  std::string compilation_unit;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;

  // same size as output_desc, describes whether
  // an output is actually a concatenation of
  // many subtensors that the fusion group produces
  std::vector<PartitionDesc> concat_desc;

  // same size as input_desc, describes whether an
  // input should be broken into subtensors (chunks)
  // to be consumed by the fusion group
  std::vector<PartitionDesc> chunk_desc;
};

} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
