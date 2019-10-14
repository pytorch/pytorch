#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
/// An ordered dictionary implementation, akin to Python's `OrderedDict`.
template <typename Key, typename Value>
class OrderedDict {
 public:
  /// A (key, value) pair.
  class Item;

  // The lifetime of an iterator is bound to the lifetime of the `OrderedDict`.
  // Further, any `insert()` operation may invalidate all iterators
  // pointing into the vector.
  using Iterator = typename std::vector<Item>::iterator;
  using ConstIterator = typename std::vector<Item>::const_iterator;

  /// Constructs the `OrderedDict` with a short description of the kinds of keys
  /// stored in the `OrderedDict`. This description is used in error messages
  /// thrown by the `OrderedDict`.
  explicit OrderedDict(std::string key_description = "Key");

  /// Copy constructs this `OrderedDict` from `other`.
  OrderedDict(const OrderedDict& other);

  /// Assigns items from `other` to this `OrderedDict`.
  OrderedDict& operator=(const OrderedDict& other);

  // NB: Move works by default, because you can move-construct vectors of const
  // values. I tried to make this noexcept (conditional on the move constructors
  // of index_ and items_ being noexcept) but the obvious spelling didn't
  // compile on Windows.
  OrderedDict(OrderedDict&& other) = default;
  OrderedDict& operator=(OrderedDict&& other) = default;

  ~OrderedDict() = default;

  /// Constructs a new `OrderedDict` and pre-populates it with the given
  /// `Item`s.
  /*implicit */ OrderedDict(std::initializer_list<Item> initializer_list);

  /// Returns the key description string the `OrderedDict` was constructed with.
  const std::string& key_description() const noexcept;

  // Element Access

  /// Returns the very first item in the `OrderedDict` and throws an exception
  /// if it is empty.
  Item& front();

  /// Returns the very first item in the `OrderedDict` and throws an exception
  /// if it is empty.
  const Item& front() const;

  /// Returns the very last item in the `OrderedDict` and throws an exception
  /// if it is empty.
  Item& back();

  /// Returns the very last item in the `OrderedDict` and throws an exception
  /// if it is empty.
  const Item& back() const;

  /// Returns the item at the `index`-th position in the `OrderedDict`. Throws
  /// an exception if the index is out of bounds.
  Item& operator[](size_t index);

  /// Returns the item at the `index`-th position in the `OrderedDict`. Throws
  /// an exception if the index is out of bounds.
  const Item& operator[](size_t index) const;

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `OrderedDict`. Use `find()` for a
  /// non-throwing way of accessing a value if it is present.
  Value& operator[](const Key& key);

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `OrderedDict`. Use `find()` for a
  /// non-throwing way of accessing a value if it is present.
  const Value& operator[](const Key& key) const;

  // Lookup

  /// Returns a pointer to the value associated with the given key, or a
  /// `nullptr` if no such key is stored in the `OrderedDict`.
  Value* find(const Key& key) noexcept;

  /// Returns a pointer to the value associated with the given key, or a
  /// `nullptr` if no such key is stored in the `OrderedDict`.
  const Value* find(const Key& key) const noexcept;

  /// Returns true if the key is present in the `OrderedDict`.
  bool contains(const Key& key) const noexcept;

  // Iterators

  /// Returns an iterator to the first item in the `OrderedDict`. Iteration is
  /// ordered.
  Iterator begin();

  /// Returns an iterator to the first item in the `OrderedDict`. Iteration is
  /// ordered.
  ConstIterator begin() const;

  /// Returns an iterator one past the last item in the `OrderedDict`.
  Iterator end();

  /// Returns an iterator one past the last item in the `OrderedDict`.
  ConstIterator end() const;

  // Capacity

  /// Returns the number of items currently stored in the `OrderedDict`.
  size_t size() const noexcept;

  /// Returns true if the `OrderedDict` contains no elements.
  bool is_empty() const noexcept;

  /// Resizes internal storage to fit at least `requested_capacity` items
  /// without requiring reallocation.
  void reserve(size_t requested_capacity);

  // Modifiers

  /// Inserts a new `(key, value)` pair into the `OrderedDict`. Throws an
  /// exception if the key is already present. If insertion is succesful,
  /// immediately returns a reference to the inserted value.
  template <typename K, typename V>
  Value& insert(K&& key, V&& value);

  /// Inserts a new `(key, value)` pair into the `OrderedDict`. Throws an
  /// exception if the key is already present. If insertion is succesful,
  /// immediately returns a reference to the inserted value.
  Value& insert(Key key, Value&& value);

  /// Inserts all items from `other` into this `OrderedDict`. If any key from
  /// `other` is already present in this `OrderedDict`, an exception is thrown.
  void update(OrderedDict&& other);

  /// Inserts all items from `other` into this `OrderedDict`. If any key from
  /// `other` is already present in this `OrderedDict`, an exception is thrown.
  void update(const OrderedDict& other);

  /// Removes the item that has `key` from this `OrderedDict` if exists and if
  /// it doesn't an exception is thrown.
  void erase(const Key& key);

  /// Removes all items from this `OrderedDict`.
  void clear();

  // Observers

  /// Returns the items stored in the `OrderedDict`.
  const std::vector<Item>& items() const noexcept;

  /// Returns a newly allocated vector and copies all keys from this
  /// `OrderedDict` into the vector.
  ::std::vector<Key> keys() const;

  /// Returns a newly allocated vector and copies all values from this
  /// `OrderedDict` into the vector.
  ::std::vector<Value> values() const;

  /// Returns a newly allocated vector and copies all keys and values from this
  /// `OrderedDict` into a vector of `std::pair<Key, Value>`.
  ::std::vector<std::pair<Key, Value>> pairs() const;

 private:
  /// A mapping from a key to an index into the `items_` vector.
  ::std::unordered_map<Key, size_t> index_;

  /// The items stored in the `OrderedDict`.
  ::std::vector<Item> items_;

  /// A description of the keys stored in the `OrderedDict`.
  ::std::string key_description_{"Key"};
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OrderedDict::Item ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Key, typename Value>
class OrderedDict<Key, Value>::Item {
 public:
  /// Constructs a new item.
  Item(Key key, Value value) : pair_(std::move(key), std::move(value)) {}

  /// Returns a reference to the value.
  Value& operator*() {
    return value();
  }

  /// Returns a reference to the value.
  const Value& operator*() const {
    return value();
  }

  /// Allows access to the value using the arrow operator.
  Value* operator->() {
    return &value();
  }

  /// Allows access to the value using the arrow operator.
  const Value* operator->() const {
    return &value();
  }

  /// Returns a reference to the key.
  const Key& key() const noexcept {
    return pair_.first;
  }

  /// Returns a reference to the value.
  Value& value() noexcept {
    return pair_.second;
  }

  /// Returns a reference to the value.
  const Value& value() const noexcept {
    return pair_.second;
  }

  /// Returns a `(key, value)` pair.
  const std::pair<Key, Value>& pair() const noexcept {
    return pair_;
  }

 private:
  /// This is stored as an std::pair because it will make Python binding a lot,
  /// lot easier.
  ::std::pair<Key, Value> pair_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OrderedDict ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Key, typename Value>
OrderedDict<Key, Value>::OrderedDict(std::string key_description)
    : key_description_(std::move(key_description)) {}

template <typename Key, typename Value>
OrderedDict<Key, Value>::OrderedDict(const OrderedDict& other)
    : index_(other.index_), key_description_(other.key_description_) {
  // Copy we have to do ourselves, because items' keys are const, so we have to
  // re-insert the items.
  for (const auto& item : other.items_) {
    items_.push_back(item);
  }
}

template <typename Key, typename Value>
OrderedDict<Key, Value>& OrderedDict<Key, Value>::operator=(
    const OrderedDict& other) {
  index_ = other.index_;
  items_.clear();
  for (auto& item : other.items_) {
    items_.push_back(item);
  }
  key_description_ = other.key_description_;
  return *this;
}

template <typename Key, typename Value>
OrderedDict<Key, Value>::OrderedDict(
    std::initializer_list<Item> initializer_list)
    : OrderedDict("Key") {
  items_.reserve(initializer_list.size());
  for (auto& item : initializer_list) {
    // Copy the key here and move it into the index.
    items_.emplace_back(item.key(), std::move(item.value()));
    index_.emplace(std::move(item.key()), size() - 1);
  }
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Iterator OrderedDict<Key, Value>::begin() {
  return items_.begin();
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::ConstIterator OrderedDict<Key, Value>::begin()
    const {
  return items_.begin();
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Iterator OrderedDict<Key, Value>::end() {
  return items_.end();
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::ConstIterator OrderedDict<Key, Value>::end()
    const {
  return items_.end();
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::front() {
  TORCH_CHECK(!items_.empty(), "Called front() on an empty OrderedDict");
  return items_.front();
}

template <typename Key, typename Value>
const typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::front()
    const {
  TORCH_CHECK(!items_.empty(), "Called front() on an empty OrderedDict");
  return items_.front();
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::back() {
  TORCH_CHECK(!items_.empty(), "Called back() on an empty OrderedDict");
  return items_.back();
}

template <typename Key, typename Value>
const typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::back()
    const {
  TORCH_CHECK(!items_.empty(), "Called back() on an empty OrderedDict");
  return items_.back();
}

template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::operator[](
    size_t index) {
  TORCH_CHECK(index < items_.size(), "Index ", index, " is out of bounds");
  return items_[index];
}

template <typename Key, typename Value>
const typename OrderedDict<Key, Value>::
    Item& OrderedDict<Key, Value>::operator[](size_t index) const {
  TORCH_CHECK(index < items_.size(), "Index ", index, " is out of bounds");
  return items_[index];
}

template <typename Key, typename Value>
Value& OrderedDict<Key, Value>::operator[](const Key& key) {
  if (auto* value = find(key)) {
    return *value;
  }
  AT_ERROR(key_description_, " '", key, "' is not defined");
}

template <typename Key, typename Value>
const Value& OrderedDict<Key, Value>::operator[](const Key& key) const {
  if (auto* value = find(key)) {
    return *value;
  }
  AT_ERROR(key_description_, " '", key, "' is not defined");
}

template <typename Key, typename Value>
template <typename K, typename V>
Value& OrderedDict<Key, Value>::insert(K&& key, V&& value) {
  TORCH_CHECK(
      index_.count(key) == 0, key_description_, " '", key, "' already defined");
  // Copy `key` here and move it into the index.
  items_.emplace_back(key, std::forward<V>(value));
  index_.emplace(std::forward<K>(key), size() - 1);
  return items_.back().value();
}

template <typename Key, typename Value>
Value& OrderedDict<Key, Value>::insert(Key key, Value&& value) {
  return insert<Key, Value>(std::move(key), std::move(value));
}

template <typename Key, typename Value>
void OrderedDict<Key, Value>::update(OrderedDict&& other) {
  reserve(size() + other.size());
  for (auto& item : other) {
    // We want to call `insert()` to prevent duplicate keys.
    insert(std::move(item.key()), std::move(item.value()));
  }
}

template <typename Key, typename Value>
void OrderedDict<Key, Value>::update(const OrderedDict& other) {
  reserve(size() + other.size());
  for (auto& item : other) {
    // We want to call `insert()` to prevent duplicate keys.
    insert(item.key(), item.value());
  }
}

template <typename Key, typename Value>
Value* OrderedDict<Key, Value>::find(const Key& key) noexcept {
  auto iterator = index_.find(key);
  if (iterator == index_.end()) {
    return nullptr;
  }
  return &items_[iterator->second].value();
}

template <typename Key, typename Value>
const Value* OrderedDict<Key, Value>::find(const Key& key) const noexcept {
  auto iterator = index_.find(key);
  if (iterator == index_.end()) {
    return nullptr;
  }
  return &items_[iterator->second].value();
}

template <typename Key, typename Value>
void OrderedDict<Key, Value>::erase(const Key& key) {
  auto it = index_.find(key);
  TORCH_CHECK(it != index_.end(), "Key '", key, "' doesn't exist");

  auto index = it->second;
  index_.erase(it);
  items_.erase(items_.begin() + index);

  for (auto& pair : index_)
    if (pair.second > index)
      --pair.second;
}

template <typename Key, typename Value>
bool OrderedDict<Key, Value>::contains(const Key& key) const noexcept {
  return find(key) != nullptr;
}

template <typename Key, typename Value>
void OrderedDict<Key, Value>::clear() {
  index_.clear();
  items_.clear();
}

template <typename Key, typename Value>
size_t OrderedDict<Key, Value>::size() const noexcept {
  return items_.size();
}

template <typename Key, typename Value>
bool OrderedDict<Key, Value>::is_empty() const noexcept {
  return items_.empty();
}

template <typename Key, typename Value>
const std::string& OrderedDict<Key, Value>::key_description() const noexcept {
  return key_description_;
}

template <typename Key, typename Value>
const std::vector<typename OrderedDict<Key, Value>::Item>& OrderedDict<
    Key,
    Value>::items() const noexcept {
  return items_;
}

template <typename Key, typename Value>
::std::vector<Key> OrderedDict<Key, Value>::keys() const {
  std::vector<Key> keys;
  keys.reserve(size());
  for (const auto& item : items_) {
    keys.push_back(item.key());
  }
  return keys;
}

template <typename Key, typename Value>
::std::vector<Value> OrderedDict<Key, Value>::values() const {
  std::vector<Value> values;
  values.reserve(size());
  for (const auto& item : items_) {
    values.push_back(item.value());
  }
  return values;
}

template <typename Key, typename Value>
::std::vector<std::pair<Key, Value>> OrderedDict<Key, Value>::pairs() const {
  std::vector<std::pair<Key, Value>> values;
  values.reserve(size());
  for (const auto& item : items_) {
    values.push_back(item.pair());
  }
  return values;
}

template <typename Key, typename Value>
void OrderedDict<Key, Value>::reserve(size_t requested_capacity) {
  index_.reserve(requested_capacity);
  items_.reserve(requested_capacity);
}

} // namespace torch
