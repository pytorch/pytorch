#pragma once
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cstdint>

namespace c10 {

template <typename T>
bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  bool is_contiguous = true;
  if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(numel, 0))) {
    return is_contiguous;
  }
  T z = 1;
  // NB: make sure we do signed arithmetic
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
    const auto& size_d = sizes[d];
    if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(size_d, 1))) {
      if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(strides[d], z))) {
        z *= size_d;
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

template <typename T>
bool _compute_channels_last_contiguous_2d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {
    case 4: {
      T expected = 1;
      for (auto& d : {1, 3, 2, 0}) {
        const auto& size_d = sizes[d];
        if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(size_d, 1))) {
          if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(strides[d], expected))) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

template <typename T>
bool _compute_channels_last_contiguous_3d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {
    case 5: {
      T expected = 1;
      for (auto& d : {1, 4, 3, 2, 0}) {
        const auto& size_d = sizes[d];
        if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(size_d, 1))) {
          if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(strides[d], expected))) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

template <typename T>
bool _compute_non_overlapping_and_dense(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  auto dim = sizes.size();
  if (dim == 1) {
    return sizes[0] < 2 || strides[0] == 1;
  }
  SmallVector<int64_t, 5> perm;
  perm.resize(dim);
  for (const auto i : c10::irange(dim)) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes[a] < 2) {
      return false;
    } else if (sizes[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });
  T require_stride = 1;
  for (const auto i : c10::irange(dim)) {
    const auto& size_perm_i = sizes[perm[i]];
    if (size_perm_i < 2) {
      return true;
    }
    if (strides[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= size_perm_i;
  }
  return true;
}

} // namespace c10
