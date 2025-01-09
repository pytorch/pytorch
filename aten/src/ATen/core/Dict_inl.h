#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/hash.h>

namespace c10 {
namespace detail {
inline bool DictKeyEqualTo::operator()(const IValue& lhs, const IValue& rhs)
    const {
  if (lhs.isTensor() && rhs.isTensor()) {
    // for tensors, we compare only by identity (following how it's done in
    // Python).
    return lhs.is(rhs);
  }
  // Otherwise, we first compare by identity for efficiency, then by value (see:
  // [container equality])
  return _fastEqualsForContainer(lhs, rhs);
}
} // namespace detail

template <class T>
decltype(auto) getTypePtr();
std::string toString(const Type& type);

namespace impl {

template <class Key, class Value>
Dict<Key, Value> toTypedDict(GenericDict dict) {
  TORCH_INTERNAL_ASSERT(
      *getTypePtr<Key>() == *dict.impl_->elementTypes.keyType,
      "Tried to cast a Dict<",
      toString(*dict.impl_->elementTypes.keyType),
      ", ",
      toString(*dict.impl_->elementTypes.valueType),
      "> to a Dict<",
      toString(*getTypePtr<Key>()),
      ", ",
      toString(*getTypePtr<Value>()),
      ">. Key types mismatch.");
  TORCH_INTERNAL_ASSERT(
      *getTypePtr<Value>() == *dict.impl_->elementTypes.valueType,
      "Tried to cast a Dict<",
      toString(*dict.impl_->elementTypes.keyType),
      ", ",
      toString(*dict.impl_->elementTypes.valueType),
      "> to a Dict<",
      toString(*getTypePtr<Key>()),
      ", ",
      toString(*getTypePtr<Value>()),
      ">. Value types mismatch.");

  return Dict<Key, Value>(std::move(dict.impl_));
}

template <class Key, class Value>
GenericDict toGenericDict(Dict<Key, Value> dict) {
  return GenericDict(std::move(dict.impl_));
}
} // namespace impl

namespace detail {

inline size_t DictKeyHash::operator()(const IValue& ivalue) const {
  if (ivalue.isInt()) {
    return std::hash<int64_t>()(ivalue.toInt());
  } else if (ivalue.isString()) {
    return std::hash<std::string_view>()(ivalue.toStringView());
  } else if (ivalue.isDouble()) {
    return std::hash<double>()(ivalue.toDouble());
  } else if (ivalue.isComplexDouble()) {
    return c10::hash<c10::complex<double>>()(ivalue.toComplexDouble());
  } else if (ivalue.isBool()) {
    return std::hash<bool>()(ivalue.toBool());
  } else if (ivalue.isTensor()) {
    return std::hash<TensorImpl*>()(ivalue.toTensor().unsafeGetTensorImpl());
  } else if (ivalue.isDevice()) {
    return std::hash<Device>()(ivalue.toDevice());
  } else {
    throw std::runtime_error(
        "Can't hash IValues with tag '" + ivalue.tagKind() + "'");
  }
}

inline intrusive_ptr<DictImpl> DictImpl::copy() const {
  return make_intrusive<DictImpl>(dict, elementTypes);
}

} // namespace detail

template <class Key, class Value>
Dict<Key, Value>::Dict()
    : Dict(make_intrusive<detail::DictImpl>(
          detail::DictImpl::dict_map_type(),
          detail::DictImpl::DictElementTypes{
              getTypePtr<Key>(),
              getTypePtr<Value>()})) {
  static_assert(
      !std::is_same_v<Key, IValue>,
      "This constructor is not valid for Dict<IValue, _>. Please use c10::impl::GenericDict(keyType, valueType) instead.");
  static_assert(
      !std::is_same_v<Value, IValue>,
      "This constructor is not valid for Dict<_, IValue>. Please use c10::impl::GenericDict(keyType, valueType) instead.");
}

template <class Key, class Value>
Dict<Key, Value>::Dict(TypePtr keyType, TypePtr valueType)
    : Dict(make_intrusive<detail::DictImpl>(
          detail::DictImpl::dict_map_type(),
          detail::DictImpl::DictElementTypes{
              std::move(keyType),
              std::move(valueType)})) {
  static_assert(
      std::is_same_v<Key, IValue>,
      "This constructor is only valid for c10::impl::GenericDict.");
  static_assert(
      std::is_same_v<Value, IValue>,
      "This constructor is only valid for c10::impl::GenericDict.");
}

template <class Key, class Value>
Dict<Key, Value>::Dict(c10::intrusive_ptr<detail::DictImpl>&& impl)
    : impl_(std::move(impl)) {}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::copy() const {
  return Dict<Key, Value>(impl_->copy());
}

template <class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::begin() const {
  return iterator{impl_->dict.begin()};
}

template <class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::end() const {
  return iterator{impl_->dict.end()};
}

template <class Key, class Value>
bool Dict<Key, Value>::empty() const {
  return impl_->dict.empty();
}

template <class Key, class Value>
typename Dict<Key, Value>::size_type Dict<Key, Value>::size() const {
  return impl_->dict.size();
}

template <class Key, class Value>
void Dict<Key, Value>::clear() const {
  impl_->dict.clear();
}

template <class Key, class Value>
template <class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert(
    Key_&& key,
    Value_&& value) const {
  static_assert(
      std::is_constructible_v<Key, Key_>,
      "Wrong type for the key argument of Dict::insert");
  static_assert(
      std::is_constructible_v<Value, Value_>,
      "Wrong type for the value argument of Dict::insert");
  auto inserted = impl_->dict.emplace(
      Key(std::forward<Key_>(key)), Value(std::forward<Value_>(value)));
  return {iterator{inserted.first}, inserted.second};
}

template <class Key, class Value>
template <class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::
    insert_or_assign(Key_&& key, Value_&& value) const {
  static_assert(
      std::is_constructible_v<Key, Key_>,
      "Wrong type for the key argument of Dict::insert_or_assign");
  static_assert(
      std::is_constructible_v<Value, Value_>,
      "Wrong type for the value argument of Dict::insert_or_assign");
  auto inserted = impl_->dict.insert_or_assign(
      Key(std::forward<Key_>(key)), Value(std::forward<Value_>(value)));
  return {iterator{inserted.first}, inserted.second};
}

template <class Key, class Value>
void Dict<Key, Value>::erase(iterator iter) const {
  impl_->dict.erase(iter.entryRef_.iterator_);
}

template <class Key, class Value>
[[nodiscard]] size_t Dict<Key, Value>::erase(const Key& key) const {
  return impl_->dict.erase(key);
}

template <class Key, class Value>
Value Dict<Key, Value>::at(const Key& key) const {
  return impl_->dict.at(key).template to<Value>();
}

template <class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::find(
    const Key& key) const {
  return iterator{impl_->dict.find(key)};
}

template <class Key, class Value>
bool Dict<Key, Value>::contains(const Key& key) const {
  return end() != find(key);
}

template <class Key, class Value>
void Dict<Key, Value>::reserve(size_type count) const {
  impl_->dict.reserve(count);
}

template <class Key, class Value>
TypePtr Dict<Key, Value>::keyType() const {
  return impl_->elementTypes.keyType;
}

template <class Key, class Value>
TypePtr Dict<Key, Value>::valueType() const {
  return impl_->elementTypes.valueType;
}
template <class Key, class Value>
void Dict<Key, Value>::unsafeSetKeyType(TypePtr t) {
  impl_->elementTypes.keyType = std::move(t);
}

template <class Key, class Value>
void Dict<Key, Value>::unsafeSetValueType(TypePtr t) {
  impl_->elementTypes.valueType = std::move(t);
}

template <class Key_, class Value_>
bool operator==(const Dict<Key_, Value_>& lhs, const Dict<Key_, Value_>& rhs) {
  // Dicts with the same identity trivially compare equal.
  if (lhs.impl_ == rhs.impl_) {
    return true;
  }

  // Otherwise compare the values
  return *lhs.impl_ == *rhs.impl_;
}

template <class Key_, class Value_>
bool operator!=(const Dict<Key_, Value_>& lhs, const Dict<Key_, Value_>& rhs) {
  return !(lhs == rhs);
}

template <class Key, class Value>
bool Dict<Key, Value>::is(const Dict& rhs) const {
  return this->impl_ == rhs.impl_;
}
} // namespace c10
