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

inline size_t DictKeyHash::operator()(const IValue& ivalue) const {
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

inline intrusive_ptr<DictImpl> DictImpl::copy() const {
  auto result = make_intrusive<DictImpl>();
  result->dict = dict;
  return result;
}

}

template<class Key, class Value>
Dict<Key, Value>::Dict()
  :Dict(make_intrusive<detail::DictImpl>()) {
}

template<class Key, class Value>
Dict<Key, Value>::Dict(Dict&& rhs) noexcept: impl_(std::move(rhs.impl_)) {
  rhs.impl_ = make_intrusive<detail::DictImpl>();
}

template<class Key, class Value>
Dict<Key, Value>::Dict(c10::intrusive_ptr<detail::DictImpl>&& impl): impl_(std::move(impl)) {}

template<class Key, class Value>
Dict<Key, Value>& Dict<Key, Value>::operator=(Dict&& rhs) noexcept {
  impl_ = std::move(rhs.impl_);
  rhs.impl_ = make_intrusive<detail::DictImpl>();
  return *this;
}

template<class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::copy() const {
  return Dict<Key, Value>(impl_->copy());
}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::begin() {
  return iterator{impl_->dict.begin()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::begin() const {
  return const_iterator{impl_->dict.begin()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::cbegin() const {
  return const_iterator{impl_->dict.cbegin()};
}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::end() {
  return iterator{impl_->dict.end()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::end() const {
  return const_iterator{impl_->dict.end()};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::cend() const {
  return const_iterator{impl_->dict.cend()};
}

template<class Key, class Value>
bool Dict<Key, Value>::empty() const {
  return impl_->dict.empty();
}

template<class Key, class Value>
typename Dict<Key, Value>::size_type Dict<Key, Value>::size() const {
  return impl_->dict.size();
}

template<class Key, class Value>
void Dict<Key, Value>::clear() {
  impl_->dict.clear();
}

template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert(Key_&& key, Value_&& value) {
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert");
  auto inserted = impl_->dict.insert(std::pair<IValue, IValue>{
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value))});
  return {iterator{inserted.first}, inserted.second};
}

template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert_or_assign(Key_&& key, Value_&& value) {
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert_or_assign");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert_or_assign");
  auto inserted = impl_->dict.insert_or_assign(
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value)));
  return {iterator{inserted.first}, inserted.second};
}

template<class Key, class Value>
void Dict<Key, Value>::erase(const_iterator iter) {
  impl_->dict.erase(iter.entryRef_.iterator_);
}

template<class Key, class Value>
C10_NODISCARD size_t Dict<Key, Value>::erase(const Key& key) {
  return impl_->dict.erase(key);
}

template<class Key, class Value>
Value Dict<Key, Value>::at(const Key& key) const {
  return impl_->dict.at(key).template to<Value>();
}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::find(const Key& key) {
  return iterator{impl_->dict.find(key)};
}

template<class Key, class Value>
typename Dict<Key, Value>::const_iterator Dict<Key, Value>::find(const Key& key) const {
  return const_iterator{impl_->dict.find(key)};
}

template<class Key, class Value>
bool Dict<Key, Value>::contains(const Key& key) const {
  return end() != find(key);
}

template<class Key, class Value>
void Dict<Key, Value>::reserve(size_type count) {
  impl_->dict.reserve(count);
}

}
