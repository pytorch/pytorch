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

// this wraps vector::iterator to make sure user code can't rely
// on it being the type of the underlying vector.
// ListT: The T in the list, i.e. int for ListPtr<int> or const int for ListPtr<const int>
// IteratorT: The T for the iterator, i.e. int for ListPtr<int>::iterator and
//   const int for ListPtr<int>::const_iterator or ListPtr<const int>::iterator or
//   ListPtr<const int>::const_iterator.
template<class ListT, class IteratorT, class Iterator, class StorageT>
class ListIterator final : public std::iterator<std::random_access_iterator_tag, ListT> {
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

  ListIterator& operator+=(typename ListPtr<ListT>::size_type offset) {
      iterator_ += offset;
      return *this;
  }

  ListIterator& operator-=(typename ListPtr<ListT>::size_type offset) {
      iterator_ -= offset;
      return *this;
  }

  ListIterator operator+(typename ListPtr<ListT>::size_type offset) const {
    return ListIterator{iterator_ + offset};
  }

  ListIterator operator-(typename ListPtr<ListT>::size_type offset) const {
    return ListIterator{iterator_ - offset};
  }

  friend typename std::iterator<std::random_access_iterator_tag, ListT>::difference_type operator-(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ - rhs.iterator_;
  }

  const IteratorT operator*() const {
      return detail::list_element_to<ListT>(*iterator_);
  }

  // the template automatically disables the operator when we are already a
  // const_iterator, because that would cause a lot of compiler warnings otherwise.
  template<class const_t_ = const IteratorT, class = guts::enable_if_t<!std::is_same<const_t_, IteratorT>::value>>
  /* implicit */ operator ListIterator<ListT, const_t_, typename detail::ListImpl<StorageT>::list_type::const_iterator, StorageT>() const
  {
      return ListIterator<ListT, const_t_, typename detail::ListImpl<StorageT>::list_type::const_iterator, StorageT> { typename detail::ListImpl<StorageT>::list_type::const_iterator { iterator_ } };
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

  friend class ListIterator<ListT, guts::remove_const_t<IteratorT>, typename detail::ListImpl<StorageT>::list_type::iterator, StorageT>;
  friend class ListPtr<ListT>;
};

template<class T> ListPtr<T> toTypedList(ListPtr<IValue> list);
template<class T> ListPtr<IValue> toGenericList(ListPtr<T> list);
const IValue* ptr_to_first_element(const ListPtr<IValue>& list);

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
public: // TODO private
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
  using iterator = impl::ListIterator<T, T, typename detail::ListImpl<StorageT>::list_type::iterator, StorageT>;
  using const_iterator = impl::ListIterator<T, const T, typename detail::ListImpl<StorageT>::list_type::const_iterator, StorageT>;
  using reverse_iterator = impl::ListIterator<T, T, typename detail::ListImpl<StorageT>::list_type::reverse_iterator, StorageT>;
  using const_reverse_iterator = impl::ListIterator<T, const T, typename detail::ListImpl<StorageT>::list_type::const_reverse_iterator, StorageT>;
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
   * Returns a reference to the element at specified location pos, with bounds checking.
   * If pos is not within the range of the container, an exception of type std::out_of_range is thrown.
   */
  const value_type operator[](size_type pos) const;

  /**
   * Assigns a new value to the element at location pos.
   */
  void set(size_type pos, const value_type& value);

  /**
   * Assigns a new value to the element at location pos.
   */
  void set(size_type pos, value_type&& value);

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  iterator begin();

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  const_iterator begin() const;

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  const_iterator cbegin() const;

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  iterator end();

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  const_iterator end() const;

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  const_iterator cend() const;

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
  void reserve(size_type new_cap);

  /**
   * Erases all elements from the container. After this call, size() returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements. Any past-the-end iterators are also invalidated.
   */
  void clear();

  /**
   * Inserts value before pos.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator insert(const_iterator pos, const T& value);

  /**
   * Inserts value before pos.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator insert(const_iterator pos, T&& value);

  /**
   * Inserts a new element into the container directly before pos.
   * The new element is constructed with the given arguments.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  template<class... Args>
  iterator emplace(const_iterator pos, Args&&... value);

  /**
   * Appends the given element value to the end of the container.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void push_back(const T& value);

  /**
   * Appends the given element value to the end of the container.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void push_back(T&& value);

  /**
   * Appends the given element value to the end of the container.
   * The new element is constructed with the given arguments.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  template<class... Args>
  void emplace_back(Args&&... args);

  /**
   * Removes the element at pos.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator erase(const_iterator pos);

  /**
   * Removes the elements in the range [first, last).
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  iterator erase(const_iterator first, const_iterator last);

  /**
   * Removes the last element of the container.
   * Calling pop_back on an empty container is undefined.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void pop_back();

  /**
   * Resizes the container to contain count elements.
   * If the current size is less than count, additional default-inserted elements are appended.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void resize(size_type count);

  /**
   * Resizes the container to contain count elements.
   * If the current size is less than count, additional copies of value are appended.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void resize(size_type count, const T& value);

protected:
  explicit ListPtr(c10::intrusive_ptr<detail::ListImpl<StorageT>>&& elements);
  template<class T_> friend ListPtr<T_> impl::toTypedList(ListPtr<IValue>);
  template<class T_> friend ListPtr<IValue> impl::toGenericList(ListPtr<T_>);
  friend const IValue* impl::ptr_to_first_element(const ListPtr<IValue>& list);
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
}

}

#include <ATen/core/List_inl.h>
