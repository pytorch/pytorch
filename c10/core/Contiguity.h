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
  if (numel == 0) {
    return true;
  }

  T expected_stride = 1;
  // NB: make sure we do signed arithmetic
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
    const auto& size_d = sizes[d];
    if (size_d == 1) {
      continue;
    }

    if (strides[d] != expected_stride) {
      return false;
    }
    expected_stride *= size_d;
  }
  return true;
}

// Return a SymBool with underlying symbolic expression that represents
// contiguity. Guaranteed not to add guards.
inline static c10::SymBool _compute_contiguous_sym(
    ArrayRef<c10::SymInt> sizes,
    ArrayRef<c10::SymInt> strides,
    const c10::SymInt& numel) {
  // If this return true, the tensor is contiguous indeed. Otherwise it could be
  // either.
  auto is_contiguous_or_false = [&]() {
    if (TORCH_GUARD_OR_FALSE(sym_eq(numel, 0))) {
      return true;
    }

    // When calculating the expected stride, we can choose to multiply
    // with max(1, size[d]) or size[d]. Regardless, this is ok for this
    // function. Why?
    // (1) If size[d] == 0, then the tensor is contiguous and if
    //     we return true or false it won't break this function.
    // (2) If size[d] is not 0, then max(1,size[d]) and size[d] are equal.
    //     Therefore, if we choose to use max(1, size[d]) or size[d] to
    //     calculate the expected stride, the result is the same.
    //
    // We symbolically check both paths to maximize the cases where this
    // function returns true. This is because make_contiguous_strides_for adds
    // the max symbolically, and in some other situations the max might not be
    // there. And we want to ensure we return true in both cases.
    c10::SymInt expected_stride = 1;
    c10::SymInt expected_stride_max = 1;
    // NB: make sure we do signed arithmetic
    for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
      if (TORCH_GUARD_OR_FALSE(sym_eq(sizes[d], 1))) {
        continue;
      }

      if (TORCH_GUARD_OR_TRUE(sym_ne(strides[d], expected_stride)) &&
          TORCH_GUARD_OR_TRUE(sym_ne(strides[d], expected_stride_max))) {
        return false;
      }
      expected_stride_max *= sizes[d].max(1);
      expected_stride *= sizes[d];
    }
    return true;
  };

  if (is_contiguous_or_false()) {
    return c10::SymBool(true);
  }

  // Build a single expression that represents contiguity and return it.
  c10::SymBool is_empty = sym_eq(numel, 0);
  c10::SymBool is_contiguous_cond = true;

  c10::SymInt expected_stride = 1;
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
    const auto& size_d = sizes[d];
    is_contiguous_cond = is_contiguous_cond.sym_and(
        size_d.sym_eq(1).sym_or(sym_eq(strides[d], expected_stride)));
    expected_stride = expected_stride * size_d;
  }
  return is_contiguous_cond.sym_or(is_empty);
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
