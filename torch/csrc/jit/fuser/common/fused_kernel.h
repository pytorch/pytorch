#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/fuser/common/annotated_graph.h"
#include "torch/csrc/jit/fuser/common/tensor_desc.h"
#include "torch/csrc/jit/fuser/common/partition_desc.h"

#include <string>
#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fuser {

struct FusedKernel {
  TH_DISALLOW_COPY_AND_ASSIGN(FusedKernel);

  FusedKernel(
    const std::string& _name
  , const std::string& _code
  , const std::vector<TensorDesc>& _input_desc
  , const std::vector<TensorDesc>& _output_desc
  , const std::vector<PartitionDesc>& _chunk_desc
  , const std::vector<PartitionDesc>& _concat_desc
  , const bool _has_random)
  : name_{_name}
  , code_{_code}
  , input_desc_{_input_desc}
  , output_desc_{_output_desc}
  , chunk_desc_{_chunk_desc}
  , concat_desc_{_concat_desc}
  , has_random_{_has_random}
  { }

  virtual ~FusedKernel() = default;

  // expects outputs to be pre-allocated
  void launch_with_tensors(
    at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

  // creates new tensors for outputs
  void launch(
    at::ArrayRef<at::Tensor> inputs
  , std::vector<at::Tensor>& outputs);

protected:
  const std::string name_;
  const std::string code_;
  const std::vector<TensorDesc> input_desc_;
  const std::vector<TensorDesc> output_desc_;
  
  // same size as input_desc, describes whether an
  // input should be broken into subtensors (chunks)
  // to be consumed by the fusion group
  const std::vector<PartitionDesc> chunk_desc_;

  // same size as output_desc, describes whether
  // an output is actually a concatenation of
  // many subtensors that the fusion group produces
  const std::vector<PartitionDesc> concat_desc_;

  const bool has_random_;

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
};

} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
