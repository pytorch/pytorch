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
  template<class T, class Enable = void> struct list_element_from final {};
  template<class T> struct list_element_from<T, guts::enable_if_t<std::is_same<IValue, typename ListPtr<T>::StorageT>::value>> final {
    static IValue call(const T& element) {
      return element;
    }
    static IValue call(T&& element) {
      return std::move(element);
    }
  };
  template<class T> struct list_element_from<T, guts::enable_if_t<!std::is_same<IValue, typename ListPtr<T>::StorageT>::value>> final {
    static T call(const T& element) {
      return element;
    }
    static T call(T&& element) {
      return std::move(element);
    }
  };
}

template<class T>
typename ListPtr<T>::value_type ListPtr<T>::get(size_type pos) const {
  return detail::list_element_to<T>(impl_->list.at(pos));
}

template<class T>
const typename ListPtr<T>::value_type ListPtr<T>::operator[](size_type pos) const {
  return get(pos);
}

template<class T>
void ListPtr<T>::set(size_type pos, const value_type& value) {
  impl_->list.at(pos) = detail::list_element_from<T>::call(value);
}

template<class T>
void ListPtr<T>::set(size_type pos, value_type&& value) {
  impl_->list.at(pos) = detail::list_element_from<T>::call(std::move(value));
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::begin() {
  return iterator(impl_->list.begin());
}

template<class T>
typename ListPtr<T>::const_iterator ListPtr<T>::begin() const {
  return cbegin();
}

template<class T>
typename ListPtr<T>::const_iterator ListPtr<T>::cbegin() const {
  return const_iterator(impl_->list.cbegin());
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::end() {
  return iterator(impl_->list.end());
}

template<class T>
typename ListPtr<T>::const_iterator ListPtr<T>::end() const {
  return cend();
}

template<class T>
typename ListPtr<T>::const_iterator ListPtr<T>::cend() const {
  return const_iterator(impl_->list.cend());
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
void ListPtr<T>::reserve(size_type new_cap) {
  impl_->list.reserve(new_cap);
}

template<class T>
void ListPtr<T>::clear() {
  impl_->list.clear();
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::insert(const_iterator pos, const T& value) {
  return iterator { impl_->list.insert(pos.iterator_, detail::list_element_from<T>::call(value)) };
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::insert(const_iterator pos, T&& value) {
  return iterator { impl_->list.insert(pos.iterator_, detail::list_element_from<T>::call(std::move(value))) };
}

template<class T>
template<class... Args>
typename ListPtr<T>::iterator ListPtr<T>::emplace(const_iterator pos, Args&&... value) {
  // TODO Use list_element_from?
  return iterator { impl_->list.emplace(pos.iterator_, std::forward<Args>(value)...) };
}

template<class T>
void ListPtr<T>::push_back(const T& value) {
  impl_->list.push_back(detail::list_element_from<T>::call(value));
}

template<class T>
void ListPtr<T>::push_back(T&& value) {
  impl_->list.push_back(detail::list_element_from<T>::call(std::move(value)));
}

template<class T>
template<class... Args>
void ListPtr<T>::emplace_back(Args&&... args) {
  // TODO Use list_element_from?
  impl_->list.emplace_back(std::forward<Args>(args)...);
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::erase(const_iterator pos) {
  return iterator { impl_->list.erase(pos.iterator_) };
}

template<class T>
typename ListPtr<T>::iterator ListPtr<T>::erase(const_iterator first, const_iterator last) {
  return iterator { impl_->list.erase(first.iterator_, last.iterator_) };
}

template<class T>
void ListPtr<T>::pop_back() {
  impl_->list.pop_back();
}

template<class T>
void ListPtr<T>::resize(size_type count) {
  impl_->list.resize(count, T{});
}

template<class T>
void ListPtr<T>::resize(size_type count, const T& value) {
  impl_->list.resize(count, value);
}

}
