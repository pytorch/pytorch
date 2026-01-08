#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <ostream>

namespace c10 {

enum class Layout : int8_t {
  Strided,
  Sparse,
  SparseCsr,
  Mkldnn,
  SparseCsc,
  SparseBsr,
  SparseBsc,
  Jagged,
  NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;
constexpr auto kJagged = Layout::Jagged;

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::kJagged;
using c10::kMkldnn;
using c10::kSparse;
using c10::kSparseBsc;
using c10::kSparseBsr;
using c10::kSparseCsc;
using c10::kSparseCsr;
using c10::kStrided;
using c10::Layout;
HIDDEN_NAMESPACE_END(torch, headeronly)
