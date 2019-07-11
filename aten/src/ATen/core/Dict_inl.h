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
DictPtr<Key, Value> make_dict() {
  return DictPtr<Key, Value>(make_intrusive<detail::DictImpl>());
}

template<class Key, class Value>
DictPtr<Key, Value>::DictPtr(DictPtr&& rhs) noexcept: impl_(std::move(rhs.impl_)) {
  rhs.impl_ = make_intrusive<detail::DictImpl>();
}

template<class Key, class Value>
DictPtr<Key, Value>::DictPtr(c10::intrusive_ptr<detail::DictImpl>&& impl): impl_(std::move(impl)) {}

template<class Key, class Value>
DictPtr<Key, Value>& DictPtr<Key, Value>::operator=(DictPtr&& rhs) noexcept {
  impl_ = std::move(rhs.impl_);
  rhs.impl_ = make_intrusive<detail::DictImpl>();
  return *this;
}

template<class Key, class Value>
DictPtr<Key, Value> DictPtr<Key, Value>::copy() const {
  return DictPtr<Key, Value>(impl_->copy());
}

template<class Key, class Value>
typename DictPtr<Key, Value>::iterator DictPtr<Key, Value>::begin() {
  return iterator{impl_->dict.begin()};
}

template<class Key, class Value>
typename DictPtr<Key, Value>::const_iterator DictPtr<Key, Value>::begin() const {
  return const_iterator{impl_->dict.begin()};
}

template<class Key, class Value>
typename DictPtr<Key, Value>::const_iterator DictPtr<Key, Value>::cbegin() const {
  return const_iterator{impl_->dict.cbegin()};
}

template<class Key, class Value>
typename DictPtr<Key, Value>::iterator DictPtr<Key, Value>::end() {
  return iterator{impl_->dict.end()};
}

template<class Key, class Value>
typename DictPtr<Key, Value>::const_iterator DictPtr<Key, Value>::end() const {
  return const_iterator{impl_->dict.end()};
}

template<class Key, class Value>
typename DictPtr<Key, Value>::const_iterator DictPtr<Key, Value>::cend() const {
  return const_iterator{impl_->dict.cend()};
}

template<class Key, class Value>
bool DictPtr<Key, Value>::empty() const {
  return impl_->dict.empty();
}

template<class Key, class Value>
typename DictPtr<Key, Value>::size_type DictPtr<Key, Value>::size() const {
  return impl_->dict.size();
}

template<class Key, class Value>
void DictPtr<Key, Value>::clear() {
  impl_->dict.clear();
}

template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename DictPtr<Key, Value>::iterator, bool> DictPtr<Key, Value>::insert(Key_&& key, Value_&& value) {
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of DictPtr::insert");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of DictPtr::insert");
  auto inserted = impl_->dict.insert(std::pair<IValue, IValue>{
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value))});
  return {iterator{inserted.first}, inserted.second};
}

template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename DictPtr<Key, Value>::iterator, bool> DictPtr<Key, Value>::insert_or_assign(Key_&& key, Value_&& value) {
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of DictPtr::insert_or_assign");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of DictPtr::insert_or_assign");
  auto inserted = impl_->dict.insert_or_assign(
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value)));
  return {iterator{inserted.first}, inserted.second};
}

template<class Key, class Value>
void DictPtr<Key, Value>::erase(const_iterator iter) {
  impl_->dict.erase(iter.entryRef_.iterator_);
}

template<class Key, class Value>
C10_NODISCARD size_t DictPtr<Key, Value>::erase(const Key& key) {
  return impl_->dict.erase(key);
}

template<class Key, class Value>
Value DictPtr<Key, Value>::at(const Key& key) const {
  return impl_->dict.at(key).template to<Value>();
}

template<class Key, class Value>
typename DictPtr<Key, Value>::iterator DictPtr<Key, Value>::find(const Key& key) {
  return iterator{impl_->dict.find(key)};
}

template<class Key, class Value>
typename DictPtr<Key, Value>::const_iterator DictPtr<Key, Value>::find(const Key& key) const {
  return const_iterator{impl_->dict.find(key)};
}

template<class Key, class Value>
bool DictPtr<Key, Value>::contains(const Key& key) const {
  return end() != find(key);
}

template<class Key, class Value>
void DictPtr<Key, Value>::reserve(size_type count) {
  impl_->dict.reserve(count);
}

}
