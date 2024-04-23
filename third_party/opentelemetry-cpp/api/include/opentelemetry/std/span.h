// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/version.h"

// Standard library implementation requires at least C++17 compiler.
// Older C++14 compilers may provide support for __has_include as a
// conforming extension.
#if defined __has_include
#  if __has_include(<version>)  // Check for __cpp_{feature}
#    include <version>
#    if defined(__cpp_lib_span) && __cplusplus > 201703L
#      define OPENTELEMETRY_HAVE_SPAN
#    endif
#  endif
#  if !defined(OPENTELEMETRY_HAVE_SPAN)
#    // Check for Visual Studio span
#    if defined(_MSVC_LANG) && _HAS_CXX20
#      define OPENTELEMETRY_HAVE_SPAN
#    endif
#    // Check for other compiler span implementation
#    if !defined(_MSVC_LANG) && __has_include(<span>) && __cplusplus > 201703L
// This works as long as compiler standard is set to C++20
#      define OPENTELEMETRY_HAVE_SPAN
#    endif
#  endif
#  if !__has_include(<gsl/gsl>)
#    undef HAVE_GSL
#  endif
#endif

#if !defined(OPENTELEMETRY_HAVE_SPAN)
#  if defined(HAVE_GSL)
#    include <type_traits>
// Guidelines Support Library provides an implementation of std::span
#    include <gsl/gsl>
OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
using gsl::dynamic_extent;
template <class ElementType, std::size_t Extent = gsl::dynamic_extent>
using span = gsl::span<ElementType, Extent>;
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
#    define OPENTELEMETRY_HAVE_SPAN
#  else
// No `gsl::span`, no `std::span`, fallback to `nostd::span`
#  endif

#else  // OPENTELEMETRY_HAVE_SPAN
// Using std::span (https://wg21.link/P0122R7) from Standard Library available in C++20 :
// - GCC libstdc++ 10+
// - Clang libc++ 7
// - MSVC Standard Library 19.26*
// - Apple Clang 10.0.0*
#  include <limits>
#  include <span>
OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
constexpr std::size_t dynamic_extent = (std::numeric_limits<std::size_t>::max)();

template <class ElementType, std::size_t Extent = nostd::dynamic_extent>
using span = std::span<ElementType, Extent>;
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
#endif  // if OPENTELEMETRY_HAVE_SPAN
