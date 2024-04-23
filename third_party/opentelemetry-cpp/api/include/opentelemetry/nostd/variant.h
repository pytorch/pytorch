// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/version.h"

#if defined(OPENTELEMETRY_STL_VERSION)
#  if OPENTELEMETRY_STL_VERSION >= 2017
#    include "opentelemetry/std/variant.h"
#    define OPENTELEMETRY_HAVE_STD_VARIANT
#  endif
#endif

#if !defined(OPENTELEMETRY_HAVE_STD_VARIANT)

#  ifndef HAVE_ABSEIL
// We use a LOCAL snapshot of Abseil that is known to compile with Visual Studio 2015.
// Header-only. Without compiling the actual Abseil binary. As Abseil moves on to new
// toolchains, it may drop support for Visual Studio 2015 in future versions.

#    if defined(__EXCEPTIONS)
#      include <exception>
OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{

class bad_variant_access : public std::exception
{
public:
  virtual const char *what() const noexcept override { return "bad_variant_access"; }
};

[[noreturn]] inline void throw_bad_variant_access()
{
  throw bad_variant_access{};
}
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
#      define THROW_BAD_VARIANT_ACCESS opentelemetry::nostd::throw_bad_variant_access()
#    else
#      define THROW_BAD_VARIANT_ACCESS std::terminate()
#    endif
#  endif

#  ifdef _MSC_VER
// Abseil variant implementation contains some benign non-impacting warnings
// that should be suppressed if compiling with Visual Studio 2017 and above.
#    pragma warning(push)
#    pragma warning(disable : 4245)  // conversion from int to const unsigned _int64
#    pragma warning(disable : 4127)  // conditional expression is constant
#  endif

#  ifdef HAVE_ABSEIL
#    include "absl/types/variant.h"
#  else
#    include "./internal/absl/types/variant.h"
#  endif

#  ifdef _MSC_VER
#    pragma warning(pop)
#  endif

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
#  ifdef HAVE_ABSEIL
using absl::bad_variant_access;
#  endif
using absl::get;
using absl::get_if;
using absl::holds_alternative;
using absl::monostate;
using absl::variant;
using absl::variant_alternative_t;
using absl::variant_size;
using absl::visit;
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE

#endif /* OPENTELEMETRY_HAVE_STD_VARIANT */
