#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/TypeList.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/ArrayRef.h>
#include <vector>

namespace at {
class Tensor;
}
namespace c10 {
struct IValue;
template<class T> class ListPtr;
template<class T> ListPtr<T> make_list();
template<class T> ListPtr<T> make_list(ArrayRef<T> values);

namespace detail {
template<class T> T list_element_to(const T& element);
template<class T> T list_element_to(const IValue& element);
template<class T, class Enable> struct list_element_from;

template<class StorageT>
struct ListImpl final : public c10::intrusive_ptr_target {
  using list_type = std::vector<StorageT>;
  list_type list;

  intrusive_ptr<ListImpl> copy() const {
    auto result = make_intrusive<ListImpl>();
    result->list = list;
    return result;
  }
};
}

namespace impl {

template<class T, class Iterator, class StorageT> class ListIterator;

template<class T, class Iterator, class StorageT> class ListElementReference;

template<class T, class Iterator, class StorageT>
void swap(ListElementReference<T, Iterator, StorageT>&& lhs, ListElementReference<T, Iterator, StorageT>&& rhs);

template<class T, class Iterator, class StorageT>
class ListElementReference final {
public:
  operator T() &&;

  ListElementReference& operator=(T&& new_value) &&;

  ListElementReference& operator=(const T& new_value) &&;

  // assigning another ref to this assigns the underlying value
  ListElementReference& operator=(ListElementReference&& rhs) &&;

  friend void swap<T, Iterator, StorageT>(ListElementReference&& lhs, ListElementReference&& rhs);

private:
  ListElementReference(Iterator iter)
  : iterator_(iter) {}

  ListElementReference(const ListElementReference&) = delete;
  ListElementReference& operator=(const ListElementReference&) = delete;

  // allow moving, but only our friends (i.e. the ListPtr class) can move us
  ListElementReference(ListElementReference&&) noexcept = default;
  ListElementReference& operator=(ListElementReference&& rhs) & noexcept {
    iterator_ = std::move(rhs.iterator_);
    return *this;
  }

  friend class ListPtr<T>;
  friend class ListIterator<T, Iterator, StorageT>;

  Iterator iterator_;
};

// this wraps vector::iterator to make sure user code can't rely
// on it being the type of the underlying vector.
template<class T, class Iterator, class StorageT>
class ListIterator final : public std::iterator<std::random_access_iterator_tag, T> {
public:
  explicit ListIterator() = default;
  ~ListIterator() = default;

  ListIterator(const ListIterator&) = default;
  ListIterator(ListIterator&&) noexcept = default;
  ListIterator& operator=(const ListIterator&) = default;
  ListIterator& operator=(ListIterator&&) = default;

  ListIterator& operator++() {
      ++iterator_;
      return *this;
  }

  ListIterator operator++(int) {
      ListIterator copy(*this);
      ++*this;
      return copy;
  }

  ListIterator& operator--() {
      --iterator_;
      return *this;
  }

  ListIterator operator--(int) {
      ListIterator copy(*this);
      --*this;
      return copy;
  }

  ListIterator& operator+=(typename ListPtr<T>::size_type offset) {
      iterator_ += offset;
      return *this;
  }

  ListIterator& operator-=(typename ListPtr<T>::size_type offset) {
      iterator_ -= offset;
      return *this;
  }

  ListIterator operator+(typename ListPtr<T>::size_type offset) const {
    return ListIterator{iterator_ + offset};
  }

  ListIterator operator-(typename ListPtr<T>::size_type offset) const {
    return ListIterator{iterator_ - offset};
  }

  friend typename std::iterator<std::random_access_iterator_tag, T>::difference_type operator-(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ - rhs.iterator_;
  }

  ListElementReference<T, Iterator, StorageT> operator*() const {
      return {iterator_};
  }

private:
  explicit ListIterator(Iterator iterator): iterator_(std::move(iterator)) {}

  Iterator iterator_;

  friend bool operator==(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ == rhs.iterator_;
  }

  friend bool operator!=(const ListIterator& lhs, const ListIterator& rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ < rhs.iterator_;
  }

  friend bool operator<=(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ <= rhs.iterator_;
  }

  friend bool operator>(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ > rhs.iterator_;
  }

  friend bool operator>=(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ >= rhs.iterator_;
  }

  friend class ListIterator<T, typename detail::ListImpl<StorageT>::list_type::iterator, StorageT>;
  friend class ListPtr<T>;
};

template<class T> ListPtr<T> toTypedList(ListPtr<IValue> list);
template<class T> ListPtr<IValue> toGenericList(ListPtr<T> list);
const IValue* ptr_to_first_element(const ListPtr<IValue>& list);
template<class T> ListPtr<T> toList(std::vector<T> list);
template<class T> ArrayRef<T> toArrayRef(const ListPtr<T>& list);
template<class T> std::vector<T> toVector(const ListPtr<T>& list);

}

/**
 * An object of this class stores a list of values of type T.
 *
 * This is a pointer type. After a copy, both ListPtrs
 * will share the same storage:
 *
 * > ListPtr<int> a = make_list<string>();
 * > ListPtr<int> b = a;
 * > b.push_back("three");
 * > ASSERT("three" == a.get(0));
 *
 * We use this class in the PyTorch kernel API instead of
 * std::vector<T>, because that allows us to do optimizations
 * and switch out the underlying list implementation without
 * breaking backwards compatibility for the kernel API.
 */
template<class T>
class ListPtr final {
private:
  // List of types that don't use IValue based lists
  using types_with_direct_list_implementation = guts::typelist::typelist<
    int64_t,
    double,
    bool,
    at::Tensor
  >;

  using StorageT = guts::conditional_t<
    guts::typelist::contains<types_with_direct_list_implementation, T>::value,
    T, // The types listed in types_with_direct_list_implementation store the list as std::vector<T>
    IValue  // All other types store the list as std::vector<IValue>
  >;

  // This is an intrusive_ptr because ListPtr is a pointer type.
  // Invariant: This will never be a nullptr, there will always be a valid
  // ListImpl.
  c10::intrusive_ptr<detail::ListImpl<StorageT>> impl_;

public:
  using value_type = T;
  using size_type = typename detail::ListImpl<StorageT>::list_type::size_type;
  using iterator = impl::ListIterator<T, typename detail::ListImpl<StorageT>::list_type::iterator, StorageT>;
  using reverse_iterator = impl::ListIterator<T, typename detail::ListImpl<StorageT>::list_type::reverse_iterator, StorageT>;
  using internal_value_type_test_only = StorageT;

  /**
   * Constructs an empty list.
   */
  friend ListPtr make_list<T>();

  /**
   * Constructs a list with some initial values
   */
  friend ListPtr make_list<T>(ArrayRef<T>);

  // please use make_list instead.
  ListPtr() = delete;

  ListPtr(const ListPtr&) = default;
  ListPtr& operator=(const ListPtr&) = default;
  ListPtr(ListPtr&&) noexcept;
  ListPtr& operator=(ListPtr&&) noexcept;

  /**
   * Create a new ListPtr pointing to a deep copy of the same data.
   * The ListPtr returned is a new list with separate storage.
   * Changes in it are not reflected in the original list or vice versa.
   */
  ListPtr copy() const;

  /**
   * Returns a reference to the element at specified location pos, with bounds checking.
   * If pos is not within the range of the container, an exception of type std::out_of_range is thrown.
   */
  value_type get(size_type pos) const;

  /**
   * Moves out the element at the specified location pos and returns it, with bounds checking.
   * If pos is not within the range of the container, an exception of type std::out_of_range is thrown.
   * The list contains an invalid element at position pos afterwards. Any operations
   * on it before re-setting it are invalid.
   */
  value_type extract(size_type pos) const;

  /**
   * Returns a reference to the element at specified location pos, with bounds checking.
   * If pos is not within the range of the container, an exception of type std::out_of_range is thrown.
   *
   * You cannot store the reference, but you can read it and assign new values to it:
   *
   *   ListPtr<int64_t> list = ...;
   *   list[2] = 5;
   *   int64_t v = list[1];
   */
  impl::ListElementReference<T, typename detail::ListImpl<typename ListPtr<T>::StorageT>::list_type::iterator, typename ListPtr<T>::StorageT> operator[](size_type pos) const;

  /**
   * Assigns a new value to the element at location pos.
   */
  void set(size_type pos, const value_type& value) const;

  /**
   * Assigns a new value to the element at location pos.
   */
  void set(size_type pos, value_type&& value) const;

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  iterator begin() const;

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  iterator end() const;

  /**
   * Checks if the container has no elements.
   */
  bool empty() const;

  /**
   * Returns the number of elements in the container
   */
  size_type size() const;

  /**
   * Increase the capacity of the vector to a value that's greater or equal to new_cap.
   */
  void reserve(size_type new_cap) const;

  /**
   * Erases all elements from the container. After this call, size() returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements. Any past-the-end iterators are also invalidated.
   */
  void clear() const;

  /**
   * Inserts value before pos.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator insert(iterator pos, const T& value) const;

  /**
   * Inserts value before pos.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator insert(iterator pos, T&& value) const;

  /**
   * Inserts a new element into the container directly before pos.
   * The new element is constructed with the given arguments.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  template<class... Args>
  iterator emplace(iterator pos, Args&&... value) const;

  /**
   * Appends the given element value to the end of the container.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void push_back(const T& value) const;

  /**
   * Appends the given element value to the end of the container.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void push_back(T&& value) const;

  /**
   * Appends the given element value to the end of the container.
   * The new element is constructed with the given arguments.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  template<class... Args>
  void emplace_back(Args&&... args) const;

  /**
   * Removes the element at pos.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator erase(iterator pos) const;

  /**
   * Removes the elements in the range [first, last).
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator erase(iterator first, iterator last) const;

  /**
   * Removes the last element of the container.
   * Calling pop_back on an empty container is undefined.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void pop_back() const;

  /**
   * Resizes the container to contain count elements.
   * If the current size is less than count, additional default-inserted elements are appended.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void resize(size_type count) const;

  /**
   * Resizes the container to contain count elements.
   * If the current size is less than count, additional copies of value are appended.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void resize(size_type count, const T& value) const;

protected:
  explicit ListPtr(c10::intrusive_ptr<detail::ListImpl<StorageT>>&& elements);
  template<class T_> friend ListPtr<T_> impl::toTypedList(ListPtr<IValue>);
  template<class T_> friend ListPtr<IValue> impl::toGenericList(ListPtr<T_>);
  friend const IValue* impl::ptr_to_first_element(const ListPtr<IValue>& list);
  template<class T_, class Enable> friend struct detail::list_element_from;
  template<class T_> friend ListPtr<T_> impl::toList(std::vector<T_> list);
  template<class T_> friend ArrayRef<T_> impl::toArrayRef(const ListPtr<T_>& list);
  template<class T_> friend std::vector<T_> impl::toVector(const ListPtr<T_>& list);
};

namespace impl {
// GenericListPtr is how IValue stores lists. It is, however, not part of the
// public API. Kernels should use Lists with concrete types instead
// (maybe except for some internal prim ops).
using GenericListPtr = ListPtr<IValue>;

inline GenericListPtr make_generic_list() {
  return make_list<IValue>();
}

inline GenericListPtr make_generic_list(ArrayRef<IValue> values) {
  return make_list<IValue>(values);
}

template<class T>
ListPtr<T> toTypedList(GenericListPtr list) {
  static_assert(std::is_same<IValue, typename ListPtr<T>::StorageT>::value, "Can only call toTypedList with lists that store their elements as IValues.");
  return ListPtr<T>(std::move(list.impl_));
}

template<class T>
GenericListPtr toGenericList(ListPtr<T> list) {
  static_assert(std::is_same<IValue, typename ListPtr<T>::StorageT>::value, "Can only call toGenericList with lists that store their elements as IValues.");
  return GenericListPtr(std::move(list.impl_));
}

inline const IValue* ptr_to_first_element(const GenericListPtr& list) {
  return &list.impl_->list[0];
}

template<class T>
ArrayRef<T> toArrayRef(const ListPtr<T>& list) {
  static_assert(std::is_same<T, IValue>::value || std::is_same<T, typename ListPtr<T>::StorageT>::value, "toArrayRef only works for lists that store their elements as std::vector<T>. You tried to call it for a list that stores its elements as std::vector<IValue>.");
  return list.impl_->list;
}

template<class T>
std::vector<T> toVector(const ListPtr<T>& list) {
  static_assert(std::is_same<T, IValue>::value || std::is_same<T, typename ListPtr<T>::StorageT>::value, "toVector only works for lists that store their elements as std::vector<T>. You tried to call it for a list that stores its elements as std::vector<IValue>.");
  std::vector<T> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); ++i) {
    result.push_back(list.get(i));
  }
  return result;
}

template<class T>
ListPtr<T> toList(std::vector<T> list) {
  static_assert(std::is_same<T, IValue>::value || std::is_same<T, typename ListPtr<T>::StorageT>::value, "toList only works for lists that store their elements as std::vector<T>. You tried to call it for a list that stores its elements as std::vector<IValue>.");
  ListPtr<T> result = make_list<T>();
  result.impl_->list = std::move(list);
  return result;
}

}

}

namespace torch {
  template<class T> using ListPtr = c10::ListPtr<T>;
}

#include <ATen/core/List_inl.h>
