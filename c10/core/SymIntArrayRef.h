#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

namespace c10 {
using SymIntArrayRef = ArrayRef<SymInt>;

inline at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar) {
  return IntArrayRef(reinterpret_cast<const int64_t*>(ar.data()), ar.size());
}

inline c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar) {
  for (const c10::SymInt& sci : ar) {
    if (sci.is_symbolic()) {
      return c10::nullopt;
    }
  }

  return {asIntArrayRefUnchecked(ar)};
}

inline at::IntArrayRef asIntArrayRefSlow(
    c10::SymIntArrayRef ar,
    const char* file,
    int64_t line) {
  for (const c10::SymInt& sci : ar) {
    TORCH_CHECK(
        !sci.is_symbolic(),
        file,
        ":",
        line,
        ": SymIntArrayRef expected to contain only concrete integers");
  }
  return asIntArrayRefUnchecked(ar);
}

#define C10_AS_INTARRAYREF_SLOW(a) c10::asIntArrayRefSlow(a, __FILE__, __LINE__)

// Prefer using a more semantic constructor, like
// fromIntArrayRefKnownNonNegative
inline SymIntArrayRef fromIntArrayRefUnchecked(IntArrayRef array_ref) {
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

inline SymIntArrayRef fromIntArrayRefKnownNonNegative(IntArrayRef array_ref) {
  return fromIntArrayRefUnchecked(array_ref);
}

inline SymIntArrayRef fromIntArrayRefSlow(IntArrayRef array_ref) {
  for (long i : array_ref) {
    TORCH_CHECK(
        SymInt::check_range(i),
        "IntArrayRef contains an int that cannot be represented as a SymInt: ",
        i);
  }
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

} // namespace c10
