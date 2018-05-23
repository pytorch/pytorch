#pragma once

#include <ATen/Error.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace detail {

/// A simple ordered dictionary implementation, akin to Python's `OrderedDict`.
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

  // Copy we have to do ourselves, because items' keys are const, so we have to
  // re-insert the items.
  OrderedDict(const OrderedDict& other) : index_(other.index_) {
    for (const auto& item : other.items_) {
      items_.push_back(item);
    }
  }

  OrderedDict& operator=(const OrderedDict& other) {
    index_ = other.index_;
    items_.clear();
    for (const auto& item : other.items_) {
      items_.push_back(item);
    }
    return *this;
  }

  // Move works by default, because you can move-construct vectors of const
  // values..
  OrderedDict(OrderedDict&& other) = default;
  OrderedDict& operator=(OrderedDict&& other) = default;

  ~OrderedDict() = default;

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

  template <typename Key, typename Value>
  T& insert(Key&& key, Value&& value) {
    AT_CHECK(index_.count(key) == 0, "Key '", key, "' already present");
    // Copy `key` here and move it into the index.
    items_.emplace_back(key, std::forward<Value>(value));
    index_.emplace(std::forward<Key>(key), size() - 1);
    return items_.back().value;
  }

  /// Allows calling `insert` with an initializer list for the value, e.g.
  /// `insert(key, {...})`.
  T& insert(std::string key, T&& value) {
    return insert<std::string, T>(std::move(key), std::move(value));
  }

  void update(OrderedDict&& other) {
    for (auto& item : other) {
      // We want to call `insert()` to prevent duplicate keys.
      insert(std::move(item.key), std::move(item.value));
    }
  }

  void update(const OrderedDict& other) {
    for (auto& item : other) {
      // We want to call `insert()` to prevent duplicate keys.
      insert(item.key, item.value);
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
    AT_ERROR("No such key: '", key, "'");
  }

  const T& get(const std::string& key) const {
    if (auto* value = find(key)) {
      return *value;
    }
    AT_ERROR("No such key: '", key, "'");
  }

  void clear() {
    index_.clear();
    items_.clear();
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
} // namespace detail
} // namespace torch
