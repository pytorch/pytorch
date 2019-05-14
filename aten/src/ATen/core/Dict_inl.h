#pragma once

#include <ATen/core/ivalue.h>

namespace c10 {
namespace impl {
inline bool shallowEquals(const IValue& lhs, const IValue& rhs) {
  if (lhs.isNone()) {
    return rhs.isNone();
  } else if (lhs.isInt()) {
    return rhs.isInt() && lhs.toInt() == rhs.toInt();
  } else if (lhs.isString()) {
    return rhs.isString() && lhs.toStringRef() == rhs.toStringRef();
  } else if (lhs.isDouble()) {
    return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
  } else if (lhs.isBool()) {
    return rhs.isBool() && lhs.toBool() == rhs.toBool();
  } else {
    AT_ERROR("shallowEquals(IValue, IValue) not implemented for type ", lhs.tagKind());
  }
}
}

namespace detail {

inline size_t DictHash::operator()(const IValue& ivalue) const {
  if (ivalue.isInt()) {
    return std::hash<int>()(ivalue.toInt());
  } else if (ivalue.isString()) {
    return std::hash<std::string>()(ivalue.toStringRef());
  } else if (ivalue.isDouble()) {
    return std::hash<double>()(ivalue.toDouble());
  } else if (ivalue.isBool()) {
    return std::hash<bool>()(ivalue.toBool());
  } else {
    throw std::runtime_error("Can't hash IValues with this tag");
  }
}

}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::begin() {
  return iterator{map_.begin()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::begin() const {
  return const_iterator{map_.begin()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::cbegin() const {
  return const_iterator{map_.cbegin()};
}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::end() {
  return iterator{map_.end()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::end() const {
  return const_iterator{map_.end()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::cend() const {
  return const_iterator{map_.cend()};
}

template<class Key, class Value>
bool Dict<Key, Value>::empty() const {
  return map_.empty();
}

template<class Key, class Value>
typename Dict<Key, Value>::size_type Dict<Key, Value>::size() const {
  return map_.size();
}

template<class Key, class Value>
void Dict<Key, Value>::clear() {
  map_.clear();
}

template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert(Key_&& key, Value_&& value) {
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert");
  auto inserted = map_.insert({
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value))});
  return {iterator{inserted.first}, inserted.second};
}

template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert_or_assign(Key_&& key, Value_&& value) {
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert_or_assign");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert_or_assign");
  auto inserted = map_.insert_or_assign(
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value)));
  return {iterator{inserted.first}, inserted.second};
}

template<class Key, class Value>
void Dict<Key, Value>::erase(const_iterator iter) {
  map_.erase(iter.entryRef_.iterator_);
}

template<class Key, class Value>
C10_NODISCARD size_t Dict<Key, Value>::erase(const Key& key) {
  return map_.erase(key);
}

template<class Key, class Value>
Value Dict<Key, Value>::at(const Key& key) {
  return map_.at(key).template to<Value>();
}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::find(const Key& key) {
  return iterator{map_.find(key)};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::find(const Key& key) const {
  return const_iterator{map_.find(key)};
}

template<class Key, class Value>
bool Dict<Key, Value>::contains(const Key& key) const {
  return end() != find(key);
}

template<class Key, class Value>
void Dict<Key, Value>::reserve(size_type count) {
  map_.reserve(count);
}

}
