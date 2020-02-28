#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/codegen/fuser/partition_desc.h>
#include <torch/csrc/jit/codegen/fuser/tensor_desc.h>
#include <torch/csrc/utils/disallow_copy.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

struct FusedKernel {
  TH_DISALLOW_COPY_AND_ASSIGN(FusedKernel);

  FusedKernel(
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random)
      : name_(std::move(name)),
        code_(std::move(code)),
        input_desc_(std::move(input_desc)),
        output_desc_(std::move(output_desc)),
        chunk_desc_(std::move(chunk_desc)),
        concat_desc_(std::move(concat_desc)),
        has_random_(has_random) {}

  virtual ~FusedKernel() = default;

  // arguments is a list of pointers to the arguments for the compiled CUDA/CPU
  // code.
  // The format of arguments is suitable for directly passing to a call to
  // cuLaunchKernel as the kernel arguments.
  // Currently the first argument is a pointer to numel (for passing to
  // CUDA code), and the remainder are pointers to the TensorInfo<T> structs
  // that compiled code uses to load Tensor data.
  // launch_with_tensors handles packing at::Tensors into this arguments array.
  // CPU code uses the same convension so that launch_with_tensors can be
  // shared.
  virtual void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const = 0;
  virtual at::Backend backend() const = 0;

  // Getters
  const std::string& name() const {
    return name_;
  }
  const std::string& code() const {
    return code_;
  }
  const std::vector<TensorDesc>& inputDesc() const {
    return input_desc_;
  }
  const std::vector<TensorDesc>& outputDesc() const {
    return output_desc_;
  }
  const std::vector<PartitionDesc>& chunkDesc() const {
    return chunk_desc_;
  }
  const std::vector<PartitionDesc>& concatDesc() const {
    return concat_desc_;
  }
  bool hasRandom() const {
    return has_random_;
  }

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
};

} // namespace fuser
} // namespace jit
} // namespace torch
