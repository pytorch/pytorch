#pragma once

#include <torch/error.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch { namespace detail {
template <typename T>
class OrderedDict {
 public:
  OrderedDict() = default;

  /*implicit */ OrderedDict(
      std::initializer_list<std::pair<std::string, T>> initializer_list) {
    values_.reserve(initializer_list.size());
    for (auto&& pair : initializer_list) {
      values_.push_back(std::move(pair.second));
      index_.emplace(std::move(pair.first), values_.size() - 1);
    }
  }

  T& insert(std::string name, T&& value) {
    if (index_.count(name) != 0) {
      TORCH_ERROR("Key %s already present", name.c_str());
    }
    values_.push_back(std::move(value));
    index_.emplace(std::move(name), values_.size() - 1);
    return values_.back();
  }

  void update(OrderedDict&& other) {
    for (size_t i = 0; i < other.size(); ++i) {
      insert(std::move(other.keys_[i]), std::move(other.values_[i]));
    }
  }

  T* find(const std::string& str) noexcept {
    auto iterator = index_.find(str);
    if (iterator == index_.end()) {
      return nullptr;
    }
    return &values_[iterator->second];
  }

  const T* find(const std::string& str) const noexcept {
    auto iterator = index_.find(str);
    if (iterator == index_.end()) {
      return nullptr;
    }
    return &values_[iterator->second];
  }

  T& get(const std::string& name) {
    if (auto* value = find(name)) {
      return *value;
    }
    TORCH_ERROR("No such key: %s", name.c_str());
  }

  const T& get(const std::string& name) const {
    if (auto* value = find(name)) {
      return *value;
    }
    TORCH_ERROR("No such key: %s", name.c_str());
  }

  const std::vector<std::string>& keys() const noexcept {
    return keys_;
  }

  const std::vector<T>& values() const noexcept {
    return values_;
  }

  std::vector<T>& values() noexcept {
    return values_;
  }

  size_t size() const noexcept {
    return values_.size();
  }

  bool is_empty() const noexcept {
    return values_.empty();
  }

 private:
  std::unordered_map<std::string, size_t> index_;
  std::vector<std::string> keys_;
  std::vector<T> values_;
};
}} // namespace torch::detail
