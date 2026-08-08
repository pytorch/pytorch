/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Demangle.h"

#ifndef _MSC_VER

#if defined(__clang__)
_Pragma("GCC diagnostic push");
_Pragma("GCC diagnostic ignored \"-Wdeprecated-dynamic-exception-spec\"");
#endif
#include <cxxabi.h>
#if defined(__clang__)
_Pragma("GCC diagnostic pop");
#endif
#endif // _MSC_VER

#include <cstring>
#include <string>

namespace KINETO_NAMESPACE {

static constexpr int kMaxSymbolSize = 8192;

std::string demangle(const char* name) {
#ifndef _MSC_VER
  if (!name) {
    return "";
  }

  if (strlen(name) > kMaxSymbolSize) {
    return name;
  }

  int status = 0;
  size_t len = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, &len, &status);
  if (status != 0) {
    return name;
  }
  std::string res(demangled);
  // The returned buffer must be freed!
  free(demangled);
  return res;
#else
  // TODO: demangling on Windows
  if (!name) {
    return "";
  } else {
    return name;
  }
#endif
}

std::string demangle(const std::string& name) {
  return demangle(name.c_str());
}

} // namespace KINETO_NAMESPACE
