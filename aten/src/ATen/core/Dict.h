#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/TypeList.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct IValue;
template<class Key, class Value> class DictPtr;

/**
 * Creates an empty dict.
 */
template<class Key, class Value>
DictPtr<Key, Value> make_dict();

namespace impl {
bool shallowEquals(const IValue& lhs, const IValue& rhs);

using valid_dict_key_types = guts::typelist::typelist<
  int64_t,
  std::string,
  double,
  bool
>;
}

namespace detail {

struct DictKeyHash {
  size_t operator()(const IValue& ivalue) const;
};

struct DictKeyEqualTo {
  bool operator()(const IValue& lhs, const IValue& rhs) const {
    return impl::shallowEquals(lhs, rhs);
  }
};

struct DictImpl final : public c10::intrusive_ptr_target {
  using dict_map_type = ska::flat_hash_map<IValue, IValue, DictKeyHash, DictKeyEqualTo>;
  dict_map_type dict;

  intrusive_ptr<DictImpl> copy() const;
};

}

namespace impl {
template<class Key, class Value, class Iterator> class DictIterator;
template<class Key, class Value, class Iterator>
bool operator==(const DictIterator<Key, Value, Iterator>& lhs, const DictIterator<Key, Value, Iterator>& rhs);

/**
 * A reference to an entry in the Dict.
 * Use the `key()` and `value()` methods to read the element.
 */
template<class Key, class Value, class Iterator>
class DictEntryRef final {
private:
  static constexpr bool is_const_ref() { return std::is_const<typename Iterator::value_type>::value; }

public:
  explicit DictEntryRef(Iterator iterator)
  : iterator_(std::move(iterator)) {}

  Key key() const {
    return iterator_->first.template to<Key>();
  }

  Value value() const {
    return iterator_->second.template to<Value>();
  }

  template<class Value_>
  void setValue(Value_&& value) const {
    static_assert(!is_const_ref(), "setValue() cannot be called on const_iterator.");
    static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of setValue()");
    iterator_->second = Value(std::forward<Value_>(value));
  }

private:
  Iterator iterator_;
  friend class DictIterator<Key, Value, Iterator>;
  friend class DictPtr<Key, Value>;
  friend bool operator==<Key, Value, Iterator>(const DictIterator<Key, Value, Iterator>& lhs, const DictIterator<Key, Value, Iterator>& rhs);
};

// this wraps map_type::iterator to make sure user code can't rely
// on it being the type of the underlying map.
template<class Key, class Value, class Iterator>
class DictIterator final : public std::iterator<std::forward_iterator_tag, DictEntryRef<Key, Value, Iterator>> {
public:
  explicit DictIterator() = default;
  ~DictIterator() = default;

  DictIterator(const DictIterator&) = default;
  DictIterator(DictIterator&&) noexcept = default;
  DictIterator& operator=(const DictIterator&) = default;
  DictIterator& operator=(DictIterator&&) = default;

  DictIterator& operator++() {
      ++entryRef_.iterator_;
      return *this;
  }

  DictIterator operator++(int) {
      DictIterator copy(*this);
      ++*this;
      return copy;
  }

  const DictEntryRef<Key, Value, Iterator>& operator*() const {
      return entryRef_;
  }

  const DictEntryRef<Key, Value, Iterator>* operator->() const {
    return &entryRef_;
  }

  // the template automatically disables the operator when we are already a
  // const_iterator, because that would cause a lot of compiler warnings otherwise.
  template<class const_iterator_ = typename detail::DictImpl::dict_map_type::const_iterator, class = guts::enable_if_t<!std::is_same<const_iterator_, Iterator>::value>>
  /* implicit */ operator DictIterator<Key, Value, const_iterator_>() const
  {
      return DictIterator<Key, Value, const_iterator_> { const_iterator_ { entryRef_.iterator_ } };
  }

private:
  explicit DictIterator(Iterator iterator): entryRef_(std::move(iterator)) {}

  DictEntryRef<Key, Value, Iterator> entryRef_;

  friend class DictIterator<Key, Value, typename detail::DictImpl::dict_map_type::iterator>;
  friend class DictPtr<Key, Value>;
  friend bool operator==<Key, Value, Iterator>(const DictIterator& lhs, const DictIterator& rhs);

  // TODO We also need comparison operators <, >, <=, >=, see ListIterator.
};

template<class Key, class Value, class Iterator>
inline bool operator==(const DictIterator<Key, Value, Iterator>& lhs, const DictIterator<Key, Value, Iterator>& rhs) {
  return lhs.entryRef_.iterator_ == rhs.entryRef_.iterator_;
}

template<class Key, class Value, class Iterator>
inline bool operator!=(const DictIterator<Key, Value, Iterator>& lhs, const DictIterator<Key, Value, Iterator>& rhs) {
  return !(lhs == rhs);
}

template<class Key, class Value> DictPtr<Key, Value> toTypedDict(DictPtr<IValue, IValue> dict);
template<class Key, class Value> DictPtr<IValue, IValue> toGenericDict(DictPtr<Key, Value> dict);
}

/**
 * An object of this class stores a map from Key to Value.
 * You can create instances using the make_dict<Key, Value>() function.
 *
 * This is a pointer type. After a copy, both DictPtrs
 * will share the same storage:
 *
 * > DictPtr<int, string> a = make_dict<int, string>();
 * > DictPtr<int, string> b = a;
 * > b.insert(3, "three");
 * > ASSERT("three" == a.at(3));
 *
 * We use this class in the PyTorch kernel API because that
 * allows us to do optimizations and switch out the underlying
 * map implementation without breaking backwards compatibility
 * for the kernel API.
 */
template<class Key, class Value>
class DictPtr final {
private:
  static_assert((std::is_same<IValue, Key>::value && std::is_same<IValue, Value>::value) || guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "Invalid Key type for Dict. We only support int64_t, double, bool, and string.");

  // impl_ stores the underlying map as a ska::flat_hash_map.
  // We intentionally don't offer conversion from/to
  // ska::flat_hash_map, return references to it or something like that,
  // because such operations would get expensive if we switch out
  // the actual map implementation.
  // This is an intrusive_ptr because DictPtr is a pointer type.
  // Invariant: This will never be a nullptr, there will always be a valid
  // DictImpl.
  c10::intrusive_ptr<detail::DictImpl> impl_;

  explicit DictPtr(c10::intrusive_ptr<detail::DictImpl>&& impl);
  friend struct IValue;
  template<class K, class V> friend DictPtr<K, V> impl::toTypedDict(DictPtr<IValue, IValue>);
  template<class K, class V> friend DictPtr<IValue, IValue> impl::toGenericDict(DictPtr<K, V>);

public:
  using key_type = Key;
  using mapped_type = Value;
  using size_type = typename detail::DictImpl::dict_map_type::size_type;
  using iterator = impl::DictIterator<Key, Value, typename detail::DictImpl::dict_map_type::iterator>;
  using const_iterator = impl::DictIterator<Key, Value, typename detail::DictImpl::dict_map_type::const_iterator>;

  /**
   * Creates an empty dict.
   */
  friend DictPtr make_dict<Key, Value>();

  // please use make_dict instead
  DictPtr() = delete;

  ~DictPtr() = default;

  DictPtr(const DictPtr&) = default;
  DictPtr& operator=(const DictPtr&) = default;
  DictPtr(DictPtr&&) noexcept;
  DictPtr& operator=(DictPtr&&) noexcept;

  /**
   * Create a new DictPtr pointing to a deep copy of the same data.
   * The DictPtr returned is a new dict with separate storage.
   * Changes in it are not reflected in the original dict or vice versa.
   */
  DictPtr copy() const;

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
   * Returns the number of elements in the container.
   */
  size_type size() const;

  /**
   * Erases all elements from the container. After this call, size() returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements. May also invalidate past-the-end iterators.
   */
  void clear();

  /**
   * Inserts element(s) into the container, if the container doesn't already contain an element with an equivalent key.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return A pair consisting of an iterator to the inserted element (or to the element that prevented the insertion) and a bool denoting whether the insertion took place.
   */
  template<class Key_, class Value_>
  std::pair<iterator, bool> insert(Key_&& key, Value_&& value);

  /**
   * If an element with the given key already exists, it is overwritten with the given value.
   * Otherwise, a new element with the given key and value are inserted.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return The bool component is true if the insertion took place and false if the assignment took place. The iterator component is pointing at the element that was inserted or updated.
   */
  template<class Key_, class Value_>
  std::pair<iterator, bool> insert_or_assign(Key_&& key, Value_&& value);

  /**
   * Removes the element pointed to by iter.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   * The iterator iter must be valid and dereferenceable. Thus the end() iterator (which is valid, but is not dereferenceable) cannot be used as a value for iter.
   */
  void erase(const_iterator iter);

  /**
   * Removes the element with the given key, if it exists.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return The number of elements removed. This is either '1' if an element with the key existed, or '0' if it didn't.
   */
  C10_NODISCARD size_t erase(const Key& key);

  /**
   * Returns the mapped value of the element with key equivalent to key.
   * If no such element exists, an exception of type std::out_of_range is thrown.
   */
  Value at(const Key& key) const;

  /**
   * Finds an element with key equivalent to key.
   *
   * @return Iterator to an element with key equivalent to key.
   *         If no such element is found, past-the-end (see end()) iterator is returned.
   */
  iterator find(const Key& key);

  /**
   * Finds an element with key equivalent to key.
   *
   * @return Iterator to an element with key equivalent to key.
   *         If no such element is found, past-the-end (see end()) iterator is returned.
   */
  const_iterator find(const Key& key) const;

  /**
   * Checks if there is an element with key equivalent to key in the container.
   *
   * @return true if there is such an element, otherwise false.
   */
  bool contains(const Key& key) const;

  /**
   * Increase the capacity so that at least count elements can be stored without
   * having to reallocate or rehash.
   */
  void reserve(size_type count);
};

namespace impl {
// GenericDictPtr is how IValue stores dicts. It is, however, not part of the
// public API. Kernels should use Dicts with concrete Key, Value types instead
// (maybe except for some internal prim ops).
using GenericDictPtr = DictPtr<IValue, IValue>;

inline GenericDictPtr make_generic_dict() {
  return make_dict<IValue, IValue>();
}

template<class Key, class Value>
DictPtr<Key, Value> toTypedDict(GenericDictPtr dict) {
  return DictPtr<Key, Value>(std::move(dict.impl_));
}

template<class Key, class Value>
GenericDictPtr toGenericDict(DictPtr<Key, Value> dict) {
  return GenericDictPtr(std::move(dict.impl_));
}

using GenericDict = GenericDictPtr;
}

template<class Key, class Value> using Dict = DictPtr<Key, Value>;

}

namespace torch {
  template<class Key, class Value> using DictPtr = c10::DictPtr<Key, Value>;
  template<class Key, class Value> using Dict = DictPtr<Key, Value>;
}

#include <ATen/core/Dict_inl.h>
