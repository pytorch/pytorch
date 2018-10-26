#pragma once

#include <c10/util/Exception.h>

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
template <typename Key, typename Value>
class OrderedDict {
 public:
  struct Item {
    Item(Key key_, Value value_)
        : key(std::move(key_)), value(std::move(value_)) {}

    Value& operator*() {
      return value;
    }
    const Value& operator*() const {
      return value;
    }
    Value* operator->() {
      return &value;
    }
    const Value* operator->() const {
      return &value;
    }

    const Key key;
    Value value;
  };

  // The lifetime of an iterator is bound to the lifetime of the `OrderedDict`.
  // Further, any `insert()` operation may invalidate all iterators
  // pointing into the vector.

  using Iterator = typename std::vector<Item>::iterator;
  using ConstIterator = typename std::vector<Item>::const_iterator;

  /// Constructs the `OrderedDict` with a string that should describe the kind
  /// of value stored in this `OrderedDict`, for example 'parameter' or
  /// 'module'.
  explicit OrderedDict(std::string subject = "Key")
      : subject_(std::move(subject)) {}

  // Copy we have to do ourselves, because items' keys are const, so we have to
  // re-insert the items.
  OrderedDict(const OrderedDict& other)
      : index_(other.index_), subject_(other.subject_) {
    for (const auto& item : other.items_) {
      items_.push_back(item);
    }
  }

  OrderedDict& operator=(const OrderedDict& other) {
    index_ = other.index_;
    items_.clear();
    for (auto& item : other.items_) {
      items_.push_back(item);
    }
    subject_ = other.subject_;
    return *this;
  }

  // Move works by default, because you can move-construct vectors of const
  // values.
  // NB: I tried to make this noexcept (conditional on the move constructors of
  // index_ and items_ being noexcept) but the obvious spelling didn't compile
  // on Windows.
  OrderedDict(OrderedDict&& other) = default;
  OrderedDict& operator=(OrderedDict&& other) = default;

  ~OrderedDict() = default;

  /*implicit */ OrderedDict(std::initializer_list<Item> initializer_list)
      : OrderedDict("Key") {
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

  Value& operator[](const Key& key) {
    return get(key);
  }

  const Value& operator[](const Key& key) const {
    return get(key);
  }

  template <typename K, typename V>
  Value& insert(K&& key, V&& value) {
    AT_CHECK(index_.count(key) == 0, subject_, " '", key, "' already defined");
    // Copy `key` here and move it into the index.
    items_.emplace_back(key, std::forward<V>(value));
    index_.emplace(std::forward<K>(key), size() - 1);
    return items_.back().value;
  }

  /// Allows calling `insert` with an initializer list for the value, e.g.
  /// `insert(key, {...})`.
  Value& insert(Key key, Value&& value) {
    return insert<Key, Value>(std::move(key), std::move(value));
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

  Value* find(const Key& key) noexcept {
    auto iterator = index_.find(key);
    if (iterator == index_.end()) {
      return nullptr;
    }
    return &items_[iterator->second].value;
  }

  const Value* find(const Key& key) const noexcept {
    auto iterator = index_.find(key);
    if (iterator == index_.end()) {
      return nullptr;
    }
    return &items_[iterator->second].value;
  }

  Value& get(const Key& key) {
    if (auto* value = find(key)) {
      return *value;
    }
    AT_ERROR(subject_, " '", key, "' is not defined");
  }

  const Value& get(const Key& key) const {
    if (auto* value = find(key)) {
      return *value;
    }
    AT_ERROR(subject_, " '", key, "' is not defined");
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

  const std::string& subject() const noexcept {
    return subject_;
  }

 private:
  std::unordered_map<Key, size_t> index_;
  std::vector<Item> items_;
  std::string subject_;
};
} // namespace detail
} // namespace torch
