#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "torch/csrc/jit/assertions.h"

#include "torch/csrc/jit/fusers/cuda/tensor_desc.h"

#include <vector>

namespace torch { namespace jit { namespace cudafuser {

struct ConcatDesc {
  
  ConcatDesc() = default;
  
  ConcatDesc(const TensorDesc& desc, size_t nSubtensors, size_t dim)
  : nSubtensors{nSubtensors}, dim{dim} {
    JIT_ASSERT(nSubtensors > 1);
    std::vector<bool> cont = desc.contiguity;
    if(dim > 0) {
      // when we narrow the concatenated output
      // we make the size[dim] smaller while keeping the stride[dim] the same,
      // meaning: stride[dim - 1] != stride[dim]*size[dim]
      // so dim - 1 is no longer contiguous
      cont[dim - 1] = false;
    }
    subtensorDesc.reset(new TensorDesc(desc.scalar_type, cont));
  }

  size_t nSubtensors = 1; // == 1 for outputs that are not concats, otherwise it is the number tensors concatenated
  size_t dim = 0; // dimension along which the concat occurs
  std::unique_ptr<TensorDesc> subtensorDesc; // descriptor for the subtensor, if it exists  
};

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)