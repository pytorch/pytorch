#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>

namespace c10 {

template<class T> TypePtr getTypePtr();
std::string toString(TypePtr typePtr);

template<class T>
List<T>::List(c10::intrusive_ptr<c10::detail::ListImpl>&& elements)
: impl_(std::move(elements)) {}

template<class T>
List<T>::List()
: List(make_intrusive<c10::detail::ListImpl>(
  typename c10::detail::ListImpl::list_type(),
  getTypePtr<T>())) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType) instead.");
}

template<class T>
List<T>::List(ArrayRef<T> values)
: List(make_intrusive<c10::detail::ListImpl>(
    typename c10::detail::ListImpl::list_type(),
    getTypePtr<T>())) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType).");
  impl_->list.reserve(values.size());
  for (const T& element : values) {
    impl_->list.push_back(element);
  }
}

template<class T>
List<T>::List(std::initializer_list<T> initial_values)
: List(ArrayRef<T>(initial_values)) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType).");
}

template<class T>
List<T>::List(TypePtr elementType)
: List(make_intrusive<c10::detail::ListImpl>(
    typename c10::detail::ListImpl::list_type(),
    std::move(elementType))) {
  static_assert(std::is_same<T, IValue>::value || std::is_same<T, c10::intrusive_ptr<ivalue::Future>>::value,
                "This constructor is only valid for c10::impl::GenericList or List<Future>.");
}

namespace impl {
template<class T>
List<T> toTypedList(impl::GenericList list) {
  TORCH_INTERNAL_ASSERT(*getTypePtr<T>() == *list.impl_->elementType, "Tried to cast a List<", toString(list.impl_->elementType), "> to a List<", toString(getTypePtr<T>()), ">. Types mismatch.");
  return List<T>(std::move(list.impl_));
}

template<class T>
impl::GenericList toList(List<T> list) {
  return GenericList(std::move(list.impl_));
}
}

template<class T>
List<T>::List(List&& rhs) noexcept: impl_(std::move(rhs.impl_)) {
  rhs.impl_ = make_intrusive<c10::detail::ListImpl>(std::vector<IValue>{}, impl_->elementType);
}

template<class T>
List<T>& List<T>::operator=(List&& rhs) noexcept {
  impl_ = std::move(rhs.impl_);
  rhs.impl_ = make_intrusive<c10::detail::ListImpl>(std::vector<IValue>{}, impl_->elementType);
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
  template<class T>
  IValue list_element_from(const T& element) {
    return element;
  }
  template<class T>
  IValue list_element_from(T&& element) {
    return std::move(element);
  }
}

namespace impl {

template<class T, class Iterator>
ListElementReference<T, Iterator>::operator T() const {
  return c10::detail::list_element_to<T>(*iterator_);
}

template<class T, class Iterator>
ListElementReference<T, Iterator>& ListElementReference<T, Iterator>::operator=(T&& new_value) && {
  *iterator_ = c10::detail::list_element_from<T>(std::move(new_value));
  return *this;
}

template<class T, class Iterator>
ListElementReference<T, Iterator>& ListElementReference<T, Iterator>::operator=(const T& new_value) && {
  *iterator_ = c10::detail::list_element_from<T>(std::move(new_value));
  return *this;
}

template<class T, class Iterator>
ListElementReference<T, Iterator>& ListElementReference<T, Iterator>::operator=(ListElementReference<T, Iterator>&& rhs) && {
  *iterator_ = *rhs.iterator_;
  return *this;
}

template<class T, class Iterator>
void swap(ListElementReference<T, Iterator>&& lhs, ListElementReference<T, Iterator>&& rhs) {
  std::swap(*lhs.iterator_, *rhs.iterator_);
}

template<class T, class Iterator>
bool operator==(const ListElementReference<T, Iterator>& lhs, const T& rhs) {
  T lhs_tmp = lhs;
  return lhs_tmp == rhs;
}

template<class T, class Iterator>
inline bool operator==(const T& lhs, const ListElementReference<T, Iterator>& rhs) {
  return rhs == lhs;
}
}

template<class T>
void List<T>::set(size_type pos, const value_type& value) const {
  impl_->list.at(pos) = c10::detail::list_element_from<T>(value);
}

template<class T>
void List<T>::set(size_type pos, value_type&& value) const {
  impl_->list.at(pos) = c10::detail::list_element_from<T>(std::move(value));
}

template<class T>
typename List<T>::value_type List<T>::get(size_type pos) const {
  return c10::detail::list_element_to<T>(impl_->list.at(pos));
}

template<class T>
typename List<T>::internal_reference_type List<T>::operator[](size_type pos) const {
  static_cast<void>(impl_->list.at(pos)); // Throw the exception if it is out of range.
  return {impl_->list.begin() + pos};
}

template<class T>
typename List<T>::value_type List<T>::extract(size_type pos) const {
  auto& elem = impl_->list.at(pos);
  auto result = c10::detail::list_element_to<T>(std::move(elem));
  // Reset the list element to a T() instead of None to keep it correctly typed
  elem = c10::detail::list_element_from<T>(T{});
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
  return iterator { impl_->list.insert(pos.iterator_, c10::detail::list_element_from<T>(value)) };
}

template<class T>
typename List<T>::iterator List<T>::insert(iterator pos, T&& value) const {
  return iterator { impl_->list.insert(pos.iterator_, c10::detail::list_element_from<T>(std::move(value))) };
}

template<class T>
template<class... Args>
typename List<T>::iterator List<T>::emplace(iterator pos, Args&&... value) const {
  // TODO Use list_element_from?
  return iterator { impl_->list.emplace(pos.iterator_, std::forward<Args>(value)...) };
}

template<class T>
void List<T>::push_back(const T& value) const {
  impl_->list.push_back(c10::detail::list_element_from<T>(value));
}

template<class T>
void List<T>::push_back(T&& value) const {
  impl_->list.push_back(c10::detail::list_element_from<T>(std::move(value)));
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
  impl_->list.push_back(T(std::forward<Args>(args)...));
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
bool operator==(const List<T>& lhs, const List<T>& rhs) {
  // Lists with the same identity trivially compare equal.
  if (lhs.impl_ == rhs.impl_) {
    return true;
  }

  // Otherwise, just compare values directly.
  return *lhs.impl_ == *rhs.impl_;
}

template<class T>
bool operator!=(const List<T>& lhs, const List<T>& rhs) {
  return !(lhs == rhs);
}

template<class T>
bool List<T>::is(const List<T>& rhs) const {
  return this->impl_ == rhs.impl_;
}

template<class T>
std::vector<T> List<T>::vec() const {
  std::vector<T> result(begin(), end());
  return result;
}

template<class T>
size_t List<T>::use_count() const {
  return impl_.use_count();
}

template <class T>
TypePtr List<T>::elementType() const {
  return impl_->elementType;
}

template <class T>
void List<T>::unsafeSetElementType(TypePtr t) {
  impl_->elementType = std::move(t);
}
}
