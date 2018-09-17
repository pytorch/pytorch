#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER
#pragma once

#include "torch/csrc/jit/fusers/common/tensor_desc.h"

#include "torch/csrc/jit/assertions.h"

#include <memory>
#include <cstdint>
#include <vector>

namespace torch { namespace jit {

// Descriptor for chunk-ing an input tensor into subtensors
// OR concat-ing an output tensor from subtensors
struct PartitionDesc {
  
  PartitionDesc()
  : nSubtensors(1), dim(0) {}

  PartitionDesc(const TensorDesc& desc, size_t nSubtensors, size_t dim)
  : nSubtensors(nSubtensors), dim(dim) {
    JIT_ASSERT(nSubtensors > 1);
    std::vector<bool> cont = desc.contiguity;
    if(dim > 0) {
      // when we narrow the concatenated output/chunked input
      // we make the size[dim] smaller while keeping the stride[dim] the same,
      // meaning: stride[dim - 1] != stride[dim]*size[dim]
      // so dim - 1 is no longer contiguous
      cont[dim - 1] = false;
    }
    subtensorDesc.reset(new TensorDesc(desc.scalar_type, cont));
  }

  bool isNoop() const {
    return nSubtensors == 1;
  }

  size_t nSubtensors; // == 1 for tensors that should not be operated on via chunk/cat
  size_t dim; // dimension along which the chunk/concat occurs
  std::unique_ptr<TensorDesc> subtensorDesc; // descriptor for the subtensor, if it exists
};

} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
