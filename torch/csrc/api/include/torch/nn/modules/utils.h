#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <optional>

#include <vector>

namespace torch {
namespace nn {
namespace modules {
namespace utils {

// Reverse the order of `t` and repeat each element for `n` times.
// This can be used to translate padding arg used by Conv and Pooling modules
// to the ones used by `F::pad`.
//
// This mirrors `_reverse_repeat_tuple` in `torch/nn/modules/utils.py`.
inline std::vector<int64_t> _reverse_repeat_vector(
    at::ArrayRef<int64_t> t,
    int64_t n) {
  TORCH_INTERNAL_ASSERT(n >= 0);
  std::vector<int64_t> ret;
  ret.reserve(t.size() * n);
  for (auto rit = t.rbegin(); rit != t.rend(); ++rit) {
    for (const auto i : c10::irange(n)) {
      (void)i; // Suppress unused variable
      ret.emplace_back(*rit);
    }
  }
  return ret;
}

inline std::vector<int64_t> _list_with_default(
    torch::ArrayRef<std::optional<int64_t>> out_size,
    torch::IntArrayRef defaults) {
  TORCH_CHECK(
      defaults.size() > out_size.size(),
      "Input dimension should be at least ",
      out_size.size() + 1);
  std::vector<int64_t> ret;
  torch::IntArrayRef defaults_slice =
      defaults.slice(defaults.size() - out_size.size(), out_size.size());
  for (const auto i : c10::irange(out_size.size())) {
    auto v = out_size.at(i);
    auto d = defaults_slice.at(i);
    ret.emplace_back(v.has_value() ? v.value() : d);
  }
  return ret;
}

} // namespace utils
} // namespace modules
} // namespace nn
} // namespace torch
