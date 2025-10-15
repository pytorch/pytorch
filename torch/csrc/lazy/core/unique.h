/**
 * Unique in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/unique.h
 */

#pragma once

#include <optional>

#include <functional>
#include <set>

namespace torch::lazy {

// Helper class to allow tracking zero or more things, which should be forcibly
// be one only thing.
template <typename T, typename C = std::equal_to<T>>
class Unique {
 public:
  std::pair<bool, const T&> set(const T& value) {
    if (value_) {
      TORCH_CHECK(C()(*value_, value), "'", *value_, "' vs '", value);
      return std::pair<bool, const T&>(false, *value_);
    }
    value_ = value;
    return std::pair<bool, const T&>(true, *value_);
  }

  operator bool() const {
    return value_.has_value();
  }
  operator const T&() const {
    return *value_;
  }
  const T& operator*() const {
    return *value_;
  }
  const T* operator->() const {
    return value_.operator->();
  }

  std::set<T> AsSet() const {
    std::set<T> vset;
    if (value_.has_value()) {
      vset.insert(*value_);
    }
    return vset;
  }

 private:
  std::optional<T> value_;
};

} // namespace torch::lazy
