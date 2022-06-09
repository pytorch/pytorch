// Taken from
// https://github.com/JuliaLang/julia/blob/v1.1.0/src/support/strtod.c
#include <torch/csrc/jit/frontend/strtod.h>

#include <c10/macros/Macros.h>
#include <clocale>
#include <cstdlib>

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <xlocale.h>
#endif

// The following code is derived from the Python function _PyOS_ascii_strtod
// see http://hg.python.org/cpython/file/default/Python/pystrtod.c
//
// Copyright © 2001-2014 Python Software Foundation; All Rights Reserved
//
// The following modifications have been made:
// - Leading spaces are ignored
// - Parsing of hex floats is supported in the derived version
// - Python functions for tolower, isdigit and malloc have been replaced by the
// respective
//   C stdlib functions

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <locale>

namespace torch {
namespace jit {

#ifdef _MSC_VER
double strtod_c(const char* nptr, char** endptr) {
  static _locale_t loc = _create_locale(LC_ALL, "C");
  return _strtod_l(nptr, endptr, loc);
}
#else
double strtod_c(const char* nptr, char** endptr) {
  /// NOLINTNEXTLINE(hicpp-signed-bitwise)
  static locale_t loc = newlocale(LC_ALL_MASK, "C", nullptr);
  return strtod_l(nptr, endptr, loc);
}
#endif

float strtof_c(const char* nptr, char** endptr) {
  return (float)strtod_c(nptr, endptr);
}

} // namespace jit
} // namespace torch
