#pragma once

#include <c10/macros/Macros.h>
#include <c10/macros/Export.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/TypeList.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <c10/util/Optional.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/jit_type_base.h>

namespace c10 {
struct IValue;
template<class Key, class Value> class Dict;
struct Type;

namespace impl {

using valid_dict_key_types = guts::typelist::typelist<
  int64_t,
  std::string,
  double,
  c10::complex<double>,
  bool,
  at::Tensor
>;
}

namespace detail {

struct DictKeyHash {
  size_t operator()(const IValue& ivalue) const;
};

struct DictKeyEqualTo {
  bool operator()(const IValue& lhs, const IValue& rhs) const;
};

struct DictImpl final : public c10::intrusive_ptr_target {
  using dict_map_type = ska_ordered::order_preserving_flat_hash_map<IValue, IValue, DictKeyHash, DictKeyEqualTo>;
  struct DictElementTypes final {
    TypePtr keyType;
    TypePtr valueType;
  };

  explicit DictImpl(dict_map_type dict_, DictElementTypes elementTypes_)
  : dict(std::move(dict_))
  , elementTypes(std::move(elementTypes_)) {}
  dict_map_type dict;

  DictElementTypes elementTypes;

  intrusive_ptr<DictImpl> copy() const;
  friend TORCH_API bool operator==(const DictImpl& lhs, const DictImpl& rhs);
};

}

namespace impl {
template<class Key, class Value, class Iterator> class DictIterator;

/**
 * A reference to an entry in the Dict.
 * Use the `key()` and `value()` methods to read the element.
 */
template<class Key, class Value, class Iterator>
class DictEntryRef final {
public:
  explicit DictEntryRef(Iterator iterator)
  : iterator_(std::move(iterator)) {}

  decltype(auto) key() const {
    return iterator_->first.template to<Key>();
  }

  decltype(auto) value() const {
    return iterator_->second.template to<Value>();
  }

  template<class Value_>
  void setValue(Value_&& value) const {
    static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of setValue()");
    iterator_->second = Value(std::forward<Value_>(value));
  }

private:
  // allow copying and moving, but only our friends (i.e. the Dict class) can do
  // it. Copying/moving this reference wrapper would be too ambiguous to allow it
  // in the public API.
  DictEntryRef(const DictEntryRef&) = default;
  DictEntryRef& operator=(const DictEntryRef&) = default;
  DictEntryRef(DictEntryRef&&) noexcept = default;
  DictEntryRef& operator=(DictEntryRef&& rhs) & noexcept = default;

  Iterator iterator_;
  friend class DictIterator<Key, Value, Iterator>;
  friend class Dict<Key, Value>;
};

// this wraps map_type::iterator to make sure user code can't rely
// on it being the type of the underlying map.
template<class Key, class Value, class Iterator>
class DictIterator final {
public:
   // C++17 friendly std::iterator implementation
  using iterator_category = std::forward_iterator_tag;
  using value_type = DictEntryRef<Key, Value, Iterator>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit DictIterator() = default;
  ~DictIterator() = default;

  DictIterator(const DictIterator& rhs): entryRef_(rhs.entryRef_) {}
  DictIterator(DictIterator&& rhs) noexcept: entryRef_(std::move(rhs.entryRef_)) {}
  DictIterator& operator=(const DictIterator& rhs) {
    entryRef_ = rhs.entryRef_;
    return *this;
  }
  DictIterator& operator=(DictIterator&& rhs) noexcept {
    entryRef_ = std::move(rhs.entryRef_);
    return *this;
  }

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

  friend difference_type operator-(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.entryRef_.iterator_ - rhs.entryRef_.iterator_;
  }

private:
  explicit DictIterator(Iterator iterator): entryRef_(std::move(iterator)) {}

  const Iterator& get_iterator_() const {
    return entryRef_.iterator_;
  }

  friend bool operator==(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() == rhs.get_iterator_();
  }

  friend bool operator!=(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() != rhs.get_iterator_();
  }

  friend bool operator<(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() < rhs.get_iterator_();
  }

  friend bool operator<=(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() <= rhs.get_iterator_();
  }

  friend bool operator>(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() > rhs.get_iterator_();
  }

  friend bool operator>=(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() >= rhs.get_iterator_();
  }

  DictEntryRef<Key, Value, Iterator> entryRef_;

  friend class DictIterator<Key, Value, typename c10::detail::DictImpl::dict_map_type::iterator>;
  friend class Dict<Key, Value>;
};

template<class Key, class Value> Dict<Key, Value> toTypedDict(Dict<IValue, IValue> dict);
template<class Key, class Value> Dict<IValue, IValue> toGenericDict(Dict<Key, Value> dict);
}

/**
 * An object of this class stores a map from Key to Value.
 *
 * This is a pointer type. After a copy, both Dicts
 * will share the same storage:
 *
 * > Dict<int, string> a;
 * > Dict<int, string> b = a;
 * > b.insert(3, "three");
 * > ASSERT("three" == a.at(3));
 *
 * We use this class in the PyTorch kernel API because that
 * allows us to do optimizations and switch out the underlying
 * map implementation without breaking backwards compatibility
 * for the kernel API.
 */
template<class Key, class Value>
class Dict final {
private:
  static_assert((std::is_same_v<IValue, Key> && std::is_same_v<IValue, Value>) || guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "Invalid Key type for Dict. We only support int64_t, double, bool, and string.");

  // impl_ stores the underlying map as a ska_ordered::order_preserving_flat_hash_map.
  // We intentionally don't offer conversion from/to
  // order_preserving_flat_hash_map, return references to it or something like that,
  // because such operations would get expensive if we switch out
  // the actual map implementation.
  // This is an intrusive_ptr because Dict is a pointer type.
  // Invariant: This will never be a nullptr, there will always be a valid
  // DictImpl.
  c10::intrusive_ptr<detail::DictImpl> impl_;

  explicit Dict(c10::intrusive_ptr<detail::DictImpl>&& impl);
  friend struct IValue;
  template<class K, class V> friend Dict<K, V> impl::toTypedDict(Dict<IValue, IValue>);
  template<class K, class V> friend Dict<IValue, IValue> impl::toGenericDict(Dict<K, V>);

public:
  using key_type = Key;
  using mapped_type = Value;
  using size_type = typename detail::DictImpl::dict_map_type::size_type;
  using iterator = impl::DictIterator<Key, Value, typename detail::DictImpl::dict_map_type::iterator>;

  /**
   * Creates an empty dict.
   */
  explicit Dict();

  /**
   * Create a generic dict with runtime type information.
   * This only works for c10::impl::GenericDict and is not part of the public API
   * but only supposed to be used internally by PyTorch.
   */
  explicit Dict(TypePtr keyType, TypePtr valueType);

  ~Dict() = default;

  Dict(const Dict&) = default;
  Dict& operator=(const Dict&) = default;

  /**
   * Create a new Dict pointing to a deep copy of the same data.
   * The Dict returned is a new dict with separate storage.
   * Changes in it are not reflected in the original dict or vice versa.
   */
  Dict copy() const;

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
   * Returns the number of elements in the container.
   */
  size_type size() const;

  /**
   * Erases all elements from the container. After this call, size() returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements. May also invalidate past-the-end iterators.
   */
  void clear() const;

  /**
   * Inserts element(s) into the container, if the container doesn't already contain an element with an equivalent key.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return A pair consisting of an iterator to the inserted element (or to the element that prevented the insertion) and a bool denoting whether the insertion took place.
   */
  template<class Key_, class Value_>
  std::pair<iterator, bool> insert(Key_&& key, Value_&& value) const;

  /**
   * If an element with the given key already exists, it is overwritten with the given value.
   * Otherwise, a new element with the given key and value are inserted.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return The bool component is true if the insertion took place and false if the assignment took place. The iterator component is pointing at the element that was inserted or updated.
   */
  template<class Key_, class Value_>
  std::pair<iterator, bool> insert_or_assign(Key_&& key, Value_&& value) const;

  /**
   * Removes the element pointed to by iter.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   * The iterator iter must be valid and dereferenceable. Thus the end() iterator (which is valid, but is not dereferenceable) cannot be used as a value for iter.
   */
  void erase(iterator iter) const;

  /**
   * Removes the element with the given key, if it exists.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return The number of elements removed. This is either '1' if an element with the key existed, or '0' if it didn't.
   */
  C10_NODISCARD size_t erase(const Key& key) const;

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
  iterator find(const Key& key) const;

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
  void reserve(size_type count) const;

  /**
   * Value equality comparison. This function implements Python-like semantics for
   * equality: two dicts with the same identity (e.g. same pointer) trivially
   * compare equal, otherwise each element is compared for equality.
   */
  template <class Key_, class Value_>
  friend bool operator==(
      const Dict<Key_, Value_>& lhs,
      const Dict<Key_, Value_>& rhs);
  template <class Key_, class Value_>
  friend bool operator!=(
      const Dict<Key_, Value_>& lhs,
      const Dict<Key_, Value_>& rhs);

  /**
   * Identity comparison. Returns true if and only if `rhs` represents the same
   * Dict object as `this`.
   */
  bool is(const Dict& rhs) const;

  // private API for now because the return type will change to TypePtr
  // instead of optional<TypePtr> once types are mandatory.
  TypePtr keyType() const;
  TypePtr valueType() const;

  // [unsafe set type]
  // These functions mutate the tagged type of this dictionary in place.
  // There is no checking that the members of the dictionary are instances
  // of the new types, nor is there a check that other IValues which
  // hold references to this dictionary have the right static type.
  // This functionality is used only in the unpickler, where at
  // creation type the real type of the dictionary is unknown, but
  // then later recovered from the static type information of the
  // unpickled object.
  void unsafeSetKeyType(TypePtr t);
  void unsafeSetValueType(TypePtr t);
};

namespace impl {
// GenericDict is how IValue stores dicts. It is, however, not part of the
// public API. Kernels should use Dicts with concrete Key, Value types instead
// (maybe except for some internal prim ops).
using GenericDict = Dict<IValue, IValue>;

}
}

namespace torch {
  template<class Key, class Value> using Dict = c10::Dict<Key, Value>;
}

#include <ATen/core/Dict_inl.h>  // IWYU pragma: keep
