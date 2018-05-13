#pragma once

#include <c10/Error.h>

// These includes are not necessary, but we have it here for historical reasons...
#include <ATen/optional.h>
#include <ATen/ATenGeneral.h>

namespace at {
  using ::c10::Error;
  using ::c10::SourceLocation;
  using ::c10::demangle;
}

#define AT_ERROR(...) C10_ERROR(__VA_ARGS__)
#define AT_ASSERT(cond) C10_ASSERT(cond)
#define AT_ASSERTM(cond, ...) C10_ASSERTM(cond, __VA_ARGS__)
#define AT_CHECK(cond, ...) C10_CHECK(cond, __VA_ARGS__)
