#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <array>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace c10 {
using SymIntArrayRef = ArrayRef<SymInt>;

TORCH_API at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar);
TORCH_API at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar);
TORCH_API c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar);

// Prefer using a more semantic constructor, like
// fromIntArrayRefKnownNonNegative
inline SymIntArrayRef fromIntArrayRefUnchecked(IntArrayRef array_ref) {
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

inline SymIntArrayRef fromIntArrayRefKnownNonNegative(IntArrayRef array_ref) {
  return fromIntArrayRefUnchecked(array_ref);
}

inline SymIntArrayRef fromIntArrayRef(IntArrayRef array_ref) {
  for (size_t i = 0; i < array_ref.size(); ++i) {
    TORCH_CHECK(
        SymInt::check_range(array_ref[i]),
        "IntArrayRef contains an int that cannot be represented as a SymInt: ",
        array_ref[i]);
  }
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

} // namespace c10
