#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace c10 {
using SymIntArrayRef = ArrayRef<SymInt>;

[[noreturn]] inline void reportSymIntArrayRefToIntArrayRefError(
    c10::SymIntArrayRef ar,
    size_t problem_index,
    const char* file,
    int64_t line) {
  const bool is_symbolic = ar[problem_index].is_symbolic();
  TORCH_CHECK(
      false,
      file,
      ":",
      line,
      ": SymIntArrayRef expected to contain only concrete integers that are "
      "stored inline and can be viewed as an IntArrayRef. Found ",
      is_symbolic ? "symbolic SymInt" : "heap-allocated concrete SymInt",
      " at index ",
      problem_index,
      ": ",
      ar[problem_index],
      " in SymIntArrayRef ",
      ar,
      is_symbolic
          ? ". This commonly happens when an operator/kernel does not support "
            "symbolic shapes at this dispatch key. Common causes include "
            "calling an eager factory/kernel with a symbolic shape outside "
            "FakeTensorMode, an operator/kernel missing SymInt support, or "
            "running under a Python dispatch mode without the Python "
            "dispatcher enabled. If this is expected during fake/meta "
            "tracing, make sure FakeTensorMode and the Python dispatcher are "
            "active; otherwise specialize or guard the symbolic size before "
            "this call."
          : ". This value is concrete, but the non-owning IntArrayRef "
            "conversion used here cannot represent heap-allocated SymInt "
            "values. Use an owning conversion or guard/specialize the value "
            "before this call.");
}

inline at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar) {
  return IntArrayRef(reinterpret_cast<const int64_t*>(ar.data()), ar.size());
}

// TODO: a SymIntArrayRef containing a heap allocated large negative integer
// can actually technically be converted to an IntArrayRef... but not with
// the non-owning API we have here.  We can't reinterpet cast; we have to
// allocate another buffer and write the integers into it.  If you need it,
// we can do it.  But I don't think you need it.

inline std::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar) {
  for (const c10::SymInt& sci : ar) {
    if (sci.is_heap_allocated()) {
      return std::nullopt;
    }
  }

  return {asIntArrayRefUnchecked(ar)};
}

inline at::IntArrayRef asIntArrayRefSlow(
    c10::SymIntArrayRef ar,
    const char* file,
    int64_t line) {
  for (const auto i : c10::irange(ar.size())) {
    if (C10_UNLIKELY(ar[i].is_heap_allocated())) {
      reportSymIntArrayRefToIntArrayRefError(ar, i, file, line);
    }
  }
  return asIntArrayRefUnchecked(ar);
}

// Even slower than asIntArrayRefSlow, as it forces an allocation for a
// destination int, BUT it is able to force specialization (it never errors)
inline c10::DimVector asIntArrayRefSlowAlloc(
    c10::SymIntArrayRef ar,
    const char* file,
    int64_t line) {
  c10::DimVector res(ar.size(), 0);
  for (const auto i : c10::irange(ar.size())) {
    res[i] = ar[i].guard_int(file, line);
  }
  return res;
}

#define C10_AS_INTARRAYREF_SLOW(a) c10::asIntArrayRefSlow(a, __FILE__, __LINE__)
#define C10_AS_INTARRAYREF_SLOW_ALLOC(a) \
  c10::asIntArrayRefSlowAlloc(a, __FILE__, __LINE__)

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

inline c10::SymBool sym_equals(SymIntArrayRef LHS, SymIntArrayRef RHS) {
  if (LHS.size() != RHS.size()) {
    return c10::SymBool(false);
  }

  c10::SymBool result(true);
  for (size_t i = 0; i < RHS.size(); ++i) {
    c10::SymBool equals = sym_eq(LHS[i], RHS[i]);
    std::optional<bool> equals_bool = equals.maybe_as_bool();

    if (equals_bool.has_value() && !*equals_bool) {
      // Early return if element comparison is known to be false
      return equals;
    }
    result = result.sym_and(equals);
  }
  return result;
}

} // namespace c10
