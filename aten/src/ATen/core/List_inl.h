#pragma once

#include <ATen/core/ivalue.h>

namespace c10 {

template<class T>
ListPtr<T> make_list() {
  return ListPtr<T>(make_intrusive<detail::ListImpl<typename ListPtr<T>::StorageT>>());
}

template<class T> ListPtr<T> make_list(ArrayRef<T> values) {
  ListPtr<T> result = make_list<T>();
  result.reserve(values.size());
  for (const T& element : values) {
    result.push_back(element);
  }
  return result;
}

template<class T>
ListPtr<T>::ListPtr(ListPtr&& rhs) noexcept: impl_(std::move(rhs.impl_)) {
  rhs.impl_ = make_intrusive<detail::ListImpl<StorageT>>();
}

template<class T>
ListPtr<T>& ListPtr<T>::operator=(ListPtr&& rhs) noexcept {
  impl_ = std::move(rhs.impl_);
  rhs.impl_ = make_intrusive<detail::ListImpl<StorageT>>();
  return *this;
}

template<class T>
ListPtr<T>::ListPtr(c10::intrusive_ptr<detail::ListImpl<StorageT>>&& elements): impl_(std::move(elements)) {}

template<class T>
ListPtr<T> ListPtr<T>::copy() const {
  return ListPtr<T>(impl_->copy());
}

namespace detail {
  template<class T>
  T list_element_to(const T& element) {
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
void ListPtr<T>::set(size_type pos, const value_type& value) const {
  impl_->list.at(pos) = detail::list_element_from<T, StorageT>(value);
}

template<class T>
void ListPtr<T>::set(size_type pos, value_type&& value) const {
  impl_->list.at(pos) = detail::list_element_from<T, StorageT>(std::move(value));
}

template<class T>
typename ListPtr<T>::value_type ListPtr<T>::get(size_type pos) const {
  return detail::list_element_to<T>(impl_->list.at(pos));
}

template<class T>
typename ListPtr<T>::internal_reference_type ListPtr<T>::operator[](size_type pos) const {
  impl_->list.at(pos); // Throw the exception if it is out of range.
  return {impl_->list.begin() + pos};
}

template<class T>
typename ListPtr<T>::value_type ListPtr<T>::extract(size_type pos) const {
  auto& elem = impl_->list.at(pos);
  auto result = detail::list_element_to<T>(std::move(elem));
  elem = detail::list_element_from<T, StorageT>(T{});
  return result;
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::begin() const {
  return iterator(impl_->list.begin());
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::end() const {
  return iterator(impl_->list.end());
}

template<class T>
bool ListPtr<T>::empty() const {
  return impl_->list.empty();
}

template<class T>
typename ListPtr<T>::size_type ListPtr<T>::size() const {
  return impl_->list.size();
}

template<class T>
void ListPtr<T>::reserve(size_type new_cap) const {
  impl_->list.reserve(new_cap);
}

template<class T>
void ListPtr<T>::clear() const {
  impl_->list.clear();
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::insert(iterator pos, const T& value) const {
  return iterator { impl_->list.insert(pos.iterator_, detail::list_element_from<T, StorageT>(value)) };
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::insert(iterator pos, T&& value) const {
  return iterator { impl_->list.insert(pos.iterator_, detail::list_element_from<T, StorageT>(std::move(value))) };
}

template<class T>
template<class... Args>
typename ListPtr<T>::iterator ListPtr<T>::emplace(iterator pos, Args&&... value) const {
  // TODO Use list_element_from?
  return iterator { impl_->list.emplace(pos.iterator_, std::forward<Args>(value)...) };
}

template<class T>
void ListPtr<T>::push_back(const T& value) const {
  impl_->list.push_back(detail::list_element_from<T, StorageT>(value));
}

template<class T>
void ListPtr<T>::push_back(T&& value) const {
  impl_->list.push_back(detail::list_element_from<T, StorageT>(std::move(value)));
}

template<class T>
template<class... Args>
void ListPtr<T>::emplace_back(Args&&... args) const {
  // TODO Use list_element_from?
  impl_->list.emplace_back(std::forward<Args>(args)...);
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::erase(iterator pos) const {
  return iterator { impl_->list.erase(pos.iterator_) };
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::erase(iterator first, iterator last) const {
  return iterator { impl_->list.erase(first.iterator_, last.iterator_) };
}

template<class T>
void ListPtr<T>::pop_back() const {
  impl_->list.pop_back();
}

template<class T>
void ListPtr<T>::resize(size_type count) const {
  impl_->list.resize(count, T{});
}

template<class T>
void ListPtr<T>::resize(size_type count, const T& value) const {
  impl_->list.resize(count, value);
}

template<class T>
bool list_is_equal(const ListPtr<T>& lhs, const ListPtr<T>& rhs) {
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
size_t ListPtr<T>::use_count() const {
  return impl_.use_count();
}

}
