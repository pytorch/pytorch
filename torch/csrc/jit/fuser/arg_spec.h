#pragma once
#include <ATen/ATen.h>
#include <ATen/core/functional.h> // fmap
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/fuser/tensor_desc.h>
#include <torch/csrc/utils/hash.h>

#include <cstdint>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Describes the (runtime) arguments to a kernel.
// ArgSpecs are also used as keys to lookup instantiated kernels, so
//  they are hashable.
// Note: the device to run on is included in the arg spec because kernels
//  are compiled per-device.
struct TORCH_API ArgSpec {
  ArgSpec(
      at::TensorList inputs,
      const int _device,
      const std::vector<int64_t>& list_input_lengths)
      : descs_{c10::fmap<TensorDesc>(inputs)},
        list_input_lengths_(list_input_lengths),
        hash_code_{torch::get_hash(
            _device,
            inputs.size(),
            descs_,
            list_input_lengths)},
        device_{_device} {}

  // (Common) hash function
  static size_t hash(const ArgSpec& spec) {
    return spec.hash_code_;
  }

  // Comparators
  bool operator==(const ArgSpec& other) const {
    return (descs_ == other.descs_ && device_ == other.device_ && list_input_lengths_ == other.list_input_lengths_);
  }

  bool operator!=(const ArgSpec& spec) const {
    return !(*this == spec);
  }

  // Getters
  size_t hashCode() const {
    return hash_code_;
  }
  const std::vector<TensorDesc>& descs() const {
    return descs_;
  }
  int device() const {
    return device_;
  }

 private:
  std::vector<TensorDesc> descs_;
  std::vector<int64_t> list_input_lengths_;
  size_t hash_code_;
  int device_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
