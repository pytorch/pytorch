#pragma once

#include <ATen/core/ivalue_to.h>
#include <ATen/core/jit_type_base.h>
#include <c10/macros/Macros.h>
#include <c10/macros/Export.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/TypeList.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <vector>

namespace at {
class Tensor;
}
namespace c10 {
struct IValue;
template<class T> class List;
struct Type;

namespace detail {

struct ListImpl final : public c10::intrusive_ptr_target {
  using list_type = std::vector<IValue>;

  explicit TORCH_API ListImpl(list_type list_, TypePtr elementType_);

  list_type list;

  TypePtr elementType;

  intrusive_ptr<ListImpl> copy() const {
    return make_intrusive<ListImpl>(list, elementType);
  }
  friend TORCH_API bool operator==(const ListImpl& lhs, const ListImpl& rhs);
};
}

namespace impl {

template<class T, class Iterator> class ListIterator;

template<class T, class Iterator> class ListElementReference;

template<class T, class Iterator>
void swap(ListElementReference<T, Iterator>&& lhs, ListElementReference<T, Iterator>&& rhs);

template<class T, class Iterator>
bool operator==(const ListElementReference<T, Iterator>& lhs, const T& rhs);

template<class T, class Iterator>
bool operator==(const T& lhs, const ListElementReference<T, Iterator>& rhs);

template<class T>
struct ListElementConstReferenceTraits {
  // In the general case, we use IValue::to().
  using const_reference = typename c10::detail::ivalue_to_const_ref_overload_return<T>::type;
};

// There is no to() overload for c10::optional<std::string>.
template<>
struct ListElementConstReferenceTraits<c10::optional<std::string>> {
  using const_reference = c10::optional<std::reference_wrapper<const std::string>>;
};

template<class T, class Iterator>
class ListElementReference final {
public:
  operator std::conditional_t<
      std::is_reference<typename c10::detail::
                            ivalue_to_const_ref_overload_return<T>::type>::value,
      const T&,
      T>() const;

  ListElementReference& operator=(T&& new_value) &&;

  ListElementReference& operator=(const T& new_value) &&;

  // assigning another ref to this assigns the underlying value
  ListElementReference& operator=(ListElementReference&& rhs) &&;

  const IValue& get() const& {
    return *iterator_;
  }

  friend void swap<T, Iterator>(ListElementReference&& lhs, ListElementReference&& rhs);

  ListElementReference(const ListElementReference&) = delete;
  ListElementReference& operator=(const ListElementReference&) = delete;

private:
  ListElementReference(Iterator iter)
  : iterator_(iter) {}

  // allow moving, but only our friends (i.e. the List class) can move us
  ListElementReference(ListElementReference&&) noexcept = default;
  ListElementReference& operator=(ListElementReference&& rhs) & noexcept {
    iterator_ = std::move(rhs.iterator_);
    return *this;
  }

  friend class List<T>;
  friend class ListIterator<T, Iterator>;

  Iterator iterator_;
};

// this wraps vector::iterator to make sure user code can't rely
// on it being the type of the underlying vector.
template <class T, class Iterator>
class ListIterator final {
 public:
   // C++17 friendly std::iterator implementation
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = ListElementReference<T, Iterator>;

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

  ListIterator& operator+=(typename List<T>::size_type offset) {
      iterator_ += offset;
      return *this;
  }

  ListIterator& operator-=(typename List<T>::size_type offset) {
      iterator_ -= offset;
      return *this;
  }

  ListIterator operator+(typename List<T>::size_type offset) const {
    return ListIterator{iterator_ + offset};
  }

  ListIterator operator-(typename List<T>::size_type offset) const {
    return ListIterator{iterator_ - offset};
  }

  friend difference_type operator-(const ListIterator& lhs, const ListIterator& rhs) {
    return lhs.iterator_ - rhs.iterator_;
  }

  ListElementReference<T, Iterator> operator*() const {
    return {iterator_};
  }

  ListElementReference<T, Iterator> operator[](typename List<T>::size_type offset) const {
    return {iterator_ + offset};
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

  friend class ListIterator<T, typename c10::detail::ListImpl::list_type::iterator>;
  friend class List<T>;
};

template<class T> List<T> toTypedList(List<IValue> list);
template<class T> List<IValue> toList(List<T>&& list);
template<class T> List<IValue> toList(const List<T>& list);
const IValue* ptr_to_first_element(const List<IValue>& list);
}

/**
 * An object of this class stores a list of values of type T.
 *
 * This is a pointer type. After a copy, both Lists
 * will share the same storage:
 *
 * > List<int> a;
 * > List<int> b = a;
 * > b.push_back("three");
 * > ASSERT("three" == a.get(0));
 *
 * We use this class in the PyTorch kernel API instead of
 * std::vector<T>, because that allows us to do optimizations
 * and switch out the underlying list implementation without
 * breaking backwards compatibility for the kernel API.
 */
template<class T>
class List final {
private:
  // This is an intrusive_ptr because List is a pointer type.
  // Invariant: This will never be a nullptr, there will always be a valid
  // ListImpl.
  c10::intrusive_ptr<c10::detail::ListImpl> impl_;

  using internal_reference_type = impl::ListElementReference<T, typename c10::detail::ListImpl::list_type::iterator>;
  using internal_const_reference_type = typename impl::ListElementConstReferenceTraits<T>::const_reference;

public:
  using value_type = T;
  using size_type = typename c10::detail::ListImpl::list_type::size_type;
  using iterator = impl::ListIterator<T, typename c10::detail::ListImpl::list_type::iterator>;
  using const_iterator = impl::ListIterator<T, typename c10::detail::ListImpl::list_type::iterator>;
  using reverse_iterator = impl::ListIterator<T, typename c10::detail::ListImpl::list_type::reverse_iterator>;

  /**
   * Constructs an empty list.
   */
  explicit List();

  /**
   * Constructs a list with some initial values.
   * Example:
   *   List<int> a({2, 3, 4});
   */
  List(std::initializer_list<T> initial_values);
  explicit List(ArrayRef<T> initial_values);

  /**
   * Create a generic list with runtime type information.
   * This only works for c10::impl::GenericList and is not part of the public API
   * but only supposed to be used internally by PyTorch.
   */
  explicit List(TypePtr elementType);

  List(const List&) = default;
  List& operator=(const List&) = default;

  /**
   * Create a new List pointing to a deep copy of the same data.
   * The List returned is a new list with separate storage.
   * Changes in it are not reflected in the original list or vice versa.
   */
  List copy() const;

  /**
   * Returns the element at specified location pos, with bounds checking.
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
   *   List<int64_t> list = ...;
   *   list[2] = 5;
   *   int64_t v = list[1];
   */
  internal_const_reference_type operator[](size_type pos) const;

  internal_reference_type operator[](size_type pos);

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
   * Appends the given list to the end of the container. Uses at most one memory allocation.
   * May invalidate any references, pointers, or iterators referring to contained elements. Any past-the-end iterators may also be invalidated.
   */
  void append(List<T> lst) const;

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

  /**
   * Value equality comparison. This function implements Python-like semantics for
   * equality: two lists with the same identity (e.g. same pointer) trivially
   * compare equal, otherwise each element is compared for equality.
   */
  template <class T_>
  friend bool operator==(const List<T_>& lhs, const List<T_>& rhs);

  template <class T_>
  friend bool operator!=(const List<T_>& lhs, const List<T_>& rhs);

  /**
   * Identity comparison. Returns true if and only if `rhs` represents the same
   * List object as `this`.
   */
  bool is(const List<T>& rhs) const;

  std::vector<T> vec() const;

  /**
   * Returns the number of Lists currently pointing to this same list.
   * If this is the only instance pointing to this list, returns 1.
   */
  // TODO Test use_count
  size_t use_count() const;

  TypePtr elementType() const;

  // See [unsafe set type] for why this exists.
  void unsafeSetElementType(TypePtr t);

private:
  explicit List(c10::intrusive_ptr<c10::detail::ListImpl>&& elements);
  explicit List(const c10::intrusive_ptr<c10::detail::ListImpl>& elements);
  friend struct IValue;
  template<class T_> friend List<T_> impl::toTypedList(List<IValue>);
  template<class T_> friend List<IValue> impl::toList(List<T_>&&);
  template<class T_> friend List<IValue> impl::toList(const List<T_>&);
  friend const IValue* impl::ptr_to_first_element(const List<IValue>& list);
};

namespace impl {
// GenericList is how IValue stores lists. It is, however, not part of the
// public API. Kernels should use Lists with concrete types instead
// (maybe except for some internal prim ops).
using GenericList = List<IValue>;

const IValue* ptr_to_first_element(const GenericList& list);

}
}

namespace torch {
  template<class T> using List = c10::List<T>;
}

#include <ATen/core/List_inl.h>  // IWYU pragma: keep
