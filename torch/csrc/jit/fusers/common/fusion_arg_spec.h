#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER
#pragma once

#include "torch/csrc/jit/fusers/common/tensor_desc.h"

#include "torch/csrc/utils/functional.h" // fmap
#include "torch/csrc/utils/hash.h"

#include "ATen/ATen.h"

#include <vector>
#include <cstdint>

namespace torch { namespace jit {

struct FusionArgSpec {
  FusionArgSpec(at::TensorList inputs)
  : descs_(fmap<TensorDesc>(inputs))
  , hash_code_(torch::get_hash(inputs.size(), descs_)) {}

  bool operator==(const FusionArgSpec& spec) const {
    return hash_code_ == spec.hash_code_ && descs_ == spec.descs_;
  }

  bool operator!=(const FusionArgSpec& spec) const {
    return !(*this == spec);
  }

  static size_t hash(const FusionArgSpec& spec) {
    return spec.hash_code_;
  }

  const std::vector<TensorDesc>& descs() const {
    return descs_;
  }

private:
  std::vector<TensorDesc> descs_;
  size_t hash_code_;
};

} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
