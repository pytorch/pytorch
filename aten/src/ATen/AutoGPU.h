#pragma once

#include <ATen/Error.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/optional.h>

#include <cstddef>

namespace at {
/// RAII guard that sets the CUDA device index in its constructor, and changes
/// it back to the index that was originally active upon destruction.
///
/// The index is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_index` after construction, the
/// destructor will still reset the index to the one that was active at
/// construction time.
///
/// The index is represented by an `optional<int32_t>`. Both `nullopt` and `-1`
/// represent the default device. `nullopt` should be preferred, support for
/// `-1` is kept for legacy reasons.
struct AutoGPU {
  /// Sets the current device to the given index if it is not `nullopt`, else
  /// does nothing.
  ///
  /// NOTE: The nvcc device compiler fails with an internal
  /// compiler error when this function is defined in the header >:(
  ///
  /// > Internal Compiler Error (codegen): "there was an error in verifying the
  /// > lgenfe output!
  explicit AutoGPU(optional<int32_t> index = nullopt);

  /// Sets the current device to the given index if it is not -1, else does
  /// nothing.
  explicit AutoGPU(int32_t index) {
    set_index(index);
  }

  /// Sets the GPU to the index on which the given tensor is located.
  explicit AutoGPU(const Tensor& tensor) {
    set_index_from(tensor);
  }

  /// Sets the GPU to the index on which the first tensor in the list is
  /// located. If the list is empty, does nothing.
  explicit AutoGPU(const TensorList& tensors) {
    if (!tensors.empty()) {
      set_index_from(tensors.front());
    }
  }

  /// Resets the GPU to the index that was active at construction of the guard.
  ~AutoGPU() {
    // It should only not have a value if an index was never actually set.
    if (original_index_.has_value()) {
      // Not checking because we don't want to throw in the destructor.
      detail::DynamicCUDAInterface::set_device(*original_index_);
    }
  }

  /// Sets the GPU to the given index if it is not `nullopt` and not -1, else
  /// does nothing.
  void set_index(at::optional<int32_t> index) {
    if (!index.has_value() || *index == -1) {
      return;
    }
    if (!original_index_.has_value()) {
      original_index_ = -1;
      detail::DynamicCUDAInterface::check_status(
          detail::DynamicCUDAInterface::get_device(&original_index_.value()));
      if (*index == *original_index_) {
        return;
      }
    }
    detail::DynamicCUDAInterface::check_status(
        detail::DynamicCUDAInterface::set_device(*index));
  }

  /// If the tensor has a CUDA type, sets the GPU to the index on which the
  /// tensor is located. Otherwise does nothing.
  /// NOTE: In the .cpp file because Windows cannot compile it >:(
  void set_index_from(const Tensor& tensor);

  /// Returns the index to which the GPU will be reset upon destruction, if
  /// any.
  const at::optional<int32_t>& original_index() const noexcept {
    return original_index_;
  }

 private:
  /// The original index that was active at construction of this object.
  at::optional<int32_t> original_index_;
};
} // namespace at
