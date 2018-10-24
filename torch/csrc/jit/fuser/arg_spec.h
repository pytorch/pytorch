#pragma once

#include "ATen/ATen.h"
#include "torch/csrc/utils/functional.h" // fmap
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/fuser/tensor_desc.h"

#include <vector>
#include <cstdint>

namespace torch { namespace jit { namespace fuser {

struct ArgSpec {
  ArgSpec(at::TensorList inputs)
  : descs_(fmap<TensorDesc>(inputs))
  , hash_code_{torch::get_hash(inputs.size(), descs_)} 
  { }

  bool operator==(const ArgSpec& spec) const {
    return hash_code_ == spec.hash_code_ && descs_ == spec.descs_;
  }

  bool operator!=(const ArgSpec& spec) const {
    return !(*this == spec);
  }

  static size_t hash(const ArgSpec& spec) {
    return spec.hash_code_;
  }

  const std::vector<TensorDesc>& descs() const {
    return descs_;
  }

private:
  std::vector<TensorDesc> descs_;
  size_t hash_code_;
};

} // namespace fuser
} // namespace jit 
} // namespace torch
