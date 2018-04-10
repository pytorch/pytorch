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
  struct Item {
    Item(std::string key_, T value_)
        : key(std::move(key_)), value(std::move(value_)) {}

    T& operator*() {
      return value;
    }
    const T& operator*() const {
      return value;
    }
    T* operator->() {
      return &value;
    }
    const T* operator->() const {
      return &value;
    }

    const std::string key;
    T value;
  };

  using Iterator = typename std::vector<Item>::iterator;
  using ConstIterator = typename std::vector<Item>::const_iterator;

  OrderedDict() = default;

  /*implicit */ OrderedDict(std::initializer_list<Item> initializer_list) {
    items_.reserve(initializer_list.size());
    for (auto& item : initializer_list) {
      // Copy the key here and move it into the index.
      items_.emplace_back(item.key, std::move(item.value));
      index_.emplace(std::move(item.key), size() - 1);
    }
  }

  Iterator begin() {
    return items_.begin();
  }
  ConstIterator begin() const {
    return items_.begin();
  }

  Iterator end() {
    return items_.end();
  }
  ConstIterator end() const {
    return items_.end();
  }

  Item& front() {
    return items_.front();
  }

  const Item& front() const {
    return items_.front();
  }

  Item& back() {
    return items_.back();
  }

  const Item& back() const {
    return items_.back();
  }

  Item& operator[](size_t index) {
    return items_[index];
  }

  const Item& operator[](size_t index) const {
    return items_[index];
  }

  T& operator[](const std::string& key) {
    return get(key);
  }

  const T& operator[](const std::string& key) const {
    return get(key);
  }

  T& insert(std::string key, T&& value) {
    if (index_.count(key) != 0) {
      TORCH_ERROR("Key %s already present", key.c_str());
    }
    // Copy `key` here and move it into the index.
    items_.emplace_back(key, std::move(value));
    index_.emplace(std::move(key), size() - 1);
    return items_.back().value;
  }

  void update(OrderedDict&& other) {
    for (auto& item : other) {
      // We want to call `insert()` to prevent duplicate keys.
      insert(std::move(item.key), std::move(item.value));
    }
  }

  T* find(const std::string& str) noexcept {
    auto iterator = index_.find(str);
    if (iterator == index_.end()) {
      return nullptr;
    }
    return &items_[iterator->second].value;
  }

  const T* find(const std::string& str) const noexcept {
    auto iterator = index_.find(str);
    if (iterator == index_.end()) {
      return nullptr;
    }
    return &items_[iterator->second].value;
  }

  T& get(const std::string& key) {
    if (auto* value = find(key)) {
      return *value;
    }
    TORCH_ERROR("No such key: %s", key.c_str());
  }

  const T& get(const std::string& key) const {
    if (auto* value = find(key)) {
      return *value;
    }
    TORCH_ERROR("No such key: %s", key.c_str());
  }

  size_t size() const noexcept {
    return items_.size();
  }

  bool is_empty() const noexcept {
    return items_.empty();
  }

 private:
  std::unordered_map<std::string, size_t> index_;
  std::vector<Item> items_;
};
}} // namespace torch::detail
