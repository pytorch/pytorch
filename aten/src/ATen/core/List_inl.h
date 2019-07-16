#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>

namespace c10 {

template<class T> TypePtr getTypePtr();
std::string toString(TypePtr typePtr);

template<class T>
List<T>::List(c10::intrusive_ptr<detail::ListImpl<StorageT>>&& elements)
: impl_(std::move(elements)) {}

template<class T>
List<T>::List()
: List(make_intrusive<detail::ListImpl<typename List<T>::StorageT>>(
  typename detail::ListImpl<typename List<T>::StorageT>::list_type(),
  getTypePtr<T>())) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType) instead, or if you absolutely have to, use c10::impl::GenericList(c10::impl::deprecatedUntypedList()).");
}

template<class T>
inline List<T>::List(c10::impl::deprecatedUntypedList)
: List(make_intrusive<detail::ListImpl<IValue>>(
    typename detail::ListImpl<IValue>::list_type(),
    c10::nullopt)) {
}

template<class T>
List<T>::List(ArrayRef<T> values)
: List(make_intrusive<detail::ListImpl<typename List<T>::StorageT>>(
    typename detail::ListImpl<typename List<T>::StorageT>::list_type(),
    getTypePtr<T>())) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType) instead, or if you absolutely have to, use c10::impl::GenericList(c10::impl::deprecatedUntypedList()).");
  impl_->list.reserve(values.size());
  for (const T& element : values) {
    impl_->list.push_back(element);
  }
}

template<class T>
List<T>::List(std::initializer_list<T> initial_values)
: List(ArrayRef<T>(initial_values)) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType) instead, or if you absolutely have to, use c10::impl::GenericList(c10::impl::deprecatedUntypedList()).");
}

template<class T>
List<T>::List(TypePtr elementType)
: List(make_intrusive<detail::ListImpl<IValue>>(
    typename detail::ListImpl<IValue>::list_type(),
    std::move(elementType))) {
  static_assert(std::is_same<T, IValue>::value, "This constructor is only valid for c10::impl::GenericList.");
}

namespace impl {
template<class T>
List<T> toTypedList(impl::GenericList list) {
  static_assert(std::is_same<IValue, typename List<T>::StorageT>::value, "Can only call toTypedList with lists that store their elements as IValues.");
  if (list.impl_->elementType.has_value()) {
    TORCH_INTERNAL_ASSERT(*getTypePtr<T>() == **list.impl_->elementType, "Tried to cast a List<", toString(*list.impl_->elementType), "> to a List<", toString(getTypePtr<T>()), ">. Types mismatch.");
  }
  return List<T>(std::move(list.impl_));
}

template<class T>
impl::GenericList toGenericList(List<T> list) {
  static_assert(std::is_same<IValue, typename List<T>::StorageT>::value, "Can only call toGenericList with lists that store their elements as IValues.");
  return GenericList(std::move(list.impl_));
}
}

template<class T>
List<T>::List(List&& rhs) noexcept: impl_(std::move(rhs.impl_)) {
  rhs.impl_ = make_intrusive<detail::ListImpl<StorageT>>(std::vector<StorageT>{}, impl_->elementType);
}

template<class T>
List<T>& List<T>::operator=(List&& rhs) noexcept {
  impl_ = std::move(rhs.impl_);
  rhs.impl_ = make_intrusive<detail::ListImpl<StorageT>>(std::vector<StorageT>{}, impl_->elementType);
  return *this;
}

template<class T>
List<T> List<T>::copy() const {
  return List<T>(impl_->copy());
}

namespace detail {
  template<class T>
  T list_element_to(T element) {
    return element;
  }
  template<class T>
  T list_element_to(const IValue& element) {
    return element.template to<T>();
  }
  template<class T>
  T list_element_to(IValue&& element) {
    return std::move(element).template to<T>();
  }
  template<class T, class StorageT>
  StorageT list_element_from(const T& element) {
    return element;
  }
  template<class T, class StorageT>
  StorageT list_element_from(T&& element) {
    return std::move(element);
  }
}

namespace impl {

template<class T, class Iterator, class StorageT>
ListElementReference<T, Iterator, StorageT>::operator T() const {
  return detail::list_element_to<T>(*iterator_);
}

template<class T, class Iterator, class StorageT>
ListElementReference<T, Iterator, StorageT>& ListElementReference<T, Iterator, StorageT>::operator=(T&& new_value) && {
  *iterator_ = detail::list_element_from<T, StorageT>(std::move(new_value));
  return *this;
}

template<class T, class Iterator, class StorageT>
ListElementReference<T, Iterator, StorageT>& ListElementReference<T, Iterator, StorageT>::operator=(const T& new_value) && {
  *iterator_ = detail::list_element_from<T, StorageT>(std::move(new_value));
  return *this;
}

template<class T, class Iterator, class StorageT>
ListElementReference<T, Iterator, StorageT>& ListElementReference<T, Iterator, StorageT>::operator=(ListElementReference<T, Iterator, StorageT>&& rhs) && {
  *iterator_ = *rhs.iterator_;
  return *this;
}

template<class T, class Iterator, class StorageT>
void swap(ListElementReference<T, Iterator, StorageT>&& lhs, ListElementReference<T, Iterator, StorageT>&& rhs) {
  std::swap(*lhs.iterator_, *rhs.iterator_);
}
}

template<class T>
void List<T>::set(size_type pos, const value_type& value) const {
  impl_->list.at(pos) = detail::list_element_from<T, StorageT>(value);
}

template<class T>
void List<T>::set(size_type pos, value_type&& value) const {
  impl_->list.at(pos) = detail::list_element_from<T, StorageT>(std::move(value));
}

template<class T>
typename List<T>::value_type List<T>::get(size_type pos) const {
  return detail::list_element_to<T>(impl_->list.at(pos));
}

template<class T>
typename List<T>::internal_reference_type List<T>::operator[](size_type pos) const {
  static_cast<void>(impl_->list.at(pos)); // Throw the exception if it is out of range.
  return {impl_->list.begin() + pos};
}

template<class T>
typename List<T>::value_type List<T>::extract(size_type pos) const {
  auto& elem = impl_->list.at(pos);
  auto result = detail::list_element_to<T>(std::move(elem));
  if (std::is_same<IValue, StorageT>::value) {
    // Reset the list element to a T() instead of None to keep it correctly typed
    elem = detail::list_element_from<T, StorageT>(T{});
  }
  return result;
}

template<class T>
typename List<T>::iterator List<T>::begin() const {
  return iterator(impl_->list.begin());
}

template<class T>
typename List<T>::iterator List<T>::end() const {
  return iterator(impl_->list.end());
}

template<class T>
bool List<T>::empty() const {
  return impl_->list.empty();
}

template<class T>
typename List<T>::size_type List<T>::size() const {
  return impl_->list.size();
}

template<class T>
void List<T>::reserve(size_type new_cap) const {
  impl_->list.reserve(new_cap);
}

template<class T>
void List<T>::clear() const {
  impl_->list.clear();
}

template<class T>
typename List<T>::iterator List<T>::insert(iterator pos, const T& value) const {
  return iterator { impl_->list.insert(pos.iterator_, detail::list_element_from<T, StorageT>(value)) };
}

template<class T>
typename List<T>::iterator List<T>::insert(iterator pos, T&& value) const {
  return iterator { impl_->list.insert(pos.iterator_, detail::list_element_from<T, StorageT>(std::move(value))) };
}

template<class T>
template<class... Args>
typename List<T>::iterator List<T>::emplace(iterator pos, Args&&... value) const {
  // TODO Use list_element_from?
  return iterator { impl_->list.emplace(pos.iterator_, std::forward<Args>(value)...) };
}

template<class T>
void List<T>::push_back(const T& value) const {
  impl_->list.push_back(detail::list_element_from<T, StorageT>(value));
}

template<class T>
void List<T>::push_back(T&& value) const {
  impl_->list.push_back(detail::list_element_from<T, StorageT>(std::move(value)));
}

template<class T>
void List<T>::append(List<T> b) const {
  if (b.use_count() == 1) {
    impl_->list.insert(impl_->list.end(), make_move_iterator(b.impl_->list.begin()), make_move_iterator(b.impl_->list.end()));
  } else {
    impl_->list.insert(impl_->list.end(), b.impl_->list.begin(), b.impl_->list.end());
  }
}

template<class T>
template<class... Args>
void List<T>::emplace_back(Args&&... args) const {
  // TODO Use list_element_from?
  impl_->list.emplace_back(std::forward<Args>(args)...);
}

template<class T>
typename List<T>::iterator List<T>::erase(iterator pos) const {
  return iterator { impl_->list.erase(pos.iterator_) };
}

template<class T>
typename List<T>::iterator List<T>::erase(iterator first, iterator last) const {
  return iterator { impl_->list.erase(first.iterator_, last.iterator_) };
}

template<class T>
void List<T>::pop_back() const {
  impl_->list.pop_back();
}

template<class T>
void List<T>::resize(size_type count) const {
  impl_->list.resize(count, T{});
}

template<class T>
void List<T>::resize(size_type count, const T& value) const {
  impl_->list.resize(count, value);
}

template<class T>
bool list_is_equal(const List<T>& lhs, const List<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs.get(i) != rhs.get(i)) {
      return false;
    }
  }
  return true;
}

template<class T>
size_t List<T>::use_count() const {
  return impl_.use_count();
}

}
