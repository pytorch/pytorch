#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>
#include <utility>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// An inline integer or one owned reference to a heap node.
// Pointer encoding is private to libtorch.
// Minimum compatible version: PyTorch 2.16.
class SymInt {
 public:
  // Constructs inline zero, with no reference to own.
  SymInt() = default;

  // Treats value as a numeric integer, not an encoded pointer word.
  // Owns a new constant node below -2^62; otherwise stores the value inline.
  /* implicit */ SymInt(int64_t value) : data_(value) {
    if (is_heap_allocated()) {
      STABLE_TORCH_ERROR_CODE_CHECK(torch_symint_from_int(value, &data_));
    }
  }

  // Shares other's node, acquiring another reference if heap allocated.
  SymInt(const SymInt& other) : data_(other.data_) {
    if (is_heap_allocated()) {
      STABLE_TORCH_ERROR_CODE_CHECK(torch_symint_incref(data_));
    }
  }

  // Transfers ownership without incref and leaves other as inline zero.
  SymInt(SymInt&& other) noexcept : data_(other.release()) {}

  // Copies other, acquiring its reference and releasing the previous one.
  SymInt& operator=(const SymInt& other) {
    SymInt copy(other);
    std::swap(data_, copy.data_);
    return *this;
  }

  // Transfers ownership from other, releasing any displaced reference.
  SymInt& operator=(SymInt&& other) noexcept {
    SymInt moved(std::move(other));
    std::swap(data_, moved.data_);
    return *this;
  }

  // Releases the owned reference, if any.
  ~SymInt() {
    if (is_heap_allocated()) {
      STABLE_TORCH_ERROR_CODE_CHECK(torch_symint_decref(data_));
    }
  }

  // Borrows the word; copying it does not acquire another reference.
  int64_t get() const {
    return data_;
  }

  // Transfers ownership to the returned word and resets this value to zero.
  // No decref; the caller must adopt or decref the returned word.
  int64_t release() {
    return std::exchange(data_, 0);
  }

  // Adopts word without incref; the caller gives up its reference.
  // Use the copy constructor to share another SymInt's reference.
  static SymInt reclaim(int64_t word) {
    SymInt result;
    result.data_ = word;
    return result;
  }

 private:
  bool is_heap_allocated() const {
    return data_ < -(INT64_C(1) << 62);
  }

  int64_t data_ = 0;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
