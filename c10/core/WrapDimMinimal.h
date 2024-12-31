#pragma once

#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <cstdint>
#include <utility>

namespace c10 {

namespace detail {
// This template can only be specialized at int64_t and c10::SymInt;
// you'll get linker errors otherwise
template <typename T>
C10_API T maybe_wrap_dim_slow(T dim, const T& dim_post_expr, bool wrap_scalar);
} // namespace detail

template <typename T>
T _maybe_wrap_dim(T dim, const T& dim_post_expr, bool wrap_scalar = true) {
  // Inline the fast paths
  if (C10_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    // For SymInts, we want an explicit control flow to trigger a guard, so we
    // may as well branch too.
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  // Check edge-cases out-of-line (wrapping scalars and out-of-bounds errors)
  return c10::detail::maybe_wrap_dim_slow<T>(
      std::move(dim), dim_post_expr, wrap_scalar);
}

inline int64_t maybe_wrap_dim(
    int64_t dim,
    int64_t dim_post_expr,
    bool wrap_scalar = true) {
  return _maybe_wrap_dim(dim, dim_post_expr, wrap_scalar);
}

inline c10::SymInt maybe_wrap_dim(
    c10::SymInt dim,
    const c10::SymInt& dim_post_expr,
    bool wrap_scalar = true) {
  return _maybe_wrap_dim(std::move(dim), dim_post_expr, wrap_scalar);
}

} // namespace c10
