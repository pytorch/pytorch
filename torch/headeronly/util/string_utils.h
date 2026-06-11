#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <string>

#if !defined(FBCODE_CAFFE2) && !defined(C10_NO_DEPRECATED)

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// NOLINTNEXTLINE(misc-unused-using-decls)
using std::stod;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::stoi;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::stoll;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::stoull;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::to_string;

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
// NOLINTNEXTLINE(misc-unused-using-decls)
using torch::headeronly::stod;
// NOLINTNEXTLINE(misc-unused-using-decls)
using torch::headeronly::stoi;
// NOLINTNEXTLINE(misc-unused-using-decls)
using torch::headeronly::stoll;
// NOLINTNEXTLINE(misc-unused-using-decls)
using torch::headeronly::stoull;
// NOLINTNEXTLINE(misc-unused-using-decls)
using torch::headeronly::to_string;
} // namespace c10

#endif
