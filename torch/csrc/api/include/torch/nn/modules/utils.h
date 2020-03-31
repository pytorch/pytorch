#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

#include <vector>

namespace torch {
namespace nn {
namespace modules {
namespace utils {

// Repeat each element of `t` for `n` times.
// This can be used to translate padding arg used by Conv and Pooling modules
// to the ones used by `F::pad`.
//
// This mirrors `_repeat_tuple` in `torch/nn/modules/utils.py`.
inline std::vector<int64_t> _repeat_vector(at::ArrayRef<int64_t> t, int64_t n) {
  std::vector<int64_t> ret;
  for (int64_t elem : t) {
    for (int64_t i = 0; i < n; i++) {
      ret.emplace_back(elem);
    }
  }
  return ret;
}

inline std::vector<int64_t> _list_with_default(
  torch::ArrayRef<c10::optional<int64_t>> out_size, torch::IntArrayRef defaults) {
  TORCH_CHECK(
    defaults.size() > out_size.size(),
    "Input dimension should be at least ", out_size.size() + 1);
  std::vector<int64_t> ret;
  torch::IntArrayRef defaults_slice = defaults.slice(defaults.size() - out_size.size(), out_size.size());
  for (size_t i = 0; i < out_size.size(); i++) {
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
