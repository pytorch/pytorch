#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeTraits.h>
#include <ATen/core/ivalue_base.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {
template<class Key, class Value> class Dict;
namespace impl {
inline bool shallowEquals(const IValue& lhs, const IValue& rhs) {
  if (lhs.isNone()) {
    return rhs.isNone();
  } else if (lhs.isInt()) {
    return rhs.isInt() && lhs.toInt() == rhs.toInt();
  } else if (lhs.isString()) {
    return rhs.isString() && lhs.toStringRef() == rhs.toStringRef();
  } else if (lhs.isDouble()) {
    return rhs.isDouble() && lhs.toDouble() == rhs.toDouble();
  } else if (lhs.isBool()) {
    return rhs.isBool() && lhs.toBool() == rhs.toBool();
  } else {
    AT_ERROR("shallowEquals(IValue, IValue) not implemented for type ", lhs.tagKind());
  }
}
}

namespace detail {

struct DictHash {
  size_t operator()(const IValue& ivalue) const {
    if (ivalue.isInt()) {
      return std::hash<int>()(ivalue.toInt());
    } else if (ivalue.isString()) {
      return std::hash<std::string>()(ivalue.toStringRef());
    } else if (ivalue.isDouble()) {
      return std::hash<double>()(ivalue.toDouble());
    } else if (ivalue.isBool()) {
      return std::hash<bool>()(ivalue.toBool());
    } else {
      throw std::runtime_error("Can't hash IValues with this tag");
    }
  }
};

struct DictEqualTo {
  bool operator()(const IValue& lhs, const IValue& rhs) const {
    return impl::shallowEquals(lhs, rhs);
  }
};

using dict_map_type = ska::flat_hash_map<IValue, IValue, DictHash, DictEqualTo>;
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
  friend class Dict<Key, Value>;
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
  template<class const_iterator_ = typename detail::dict_map_type::const_iterator, class = guts::enable_if_t<!std::is_same<const_iterator_, Iterator>::value>>
  /* implicit */ operator DictIterator<Key, Value, const_iterator_>() const
  {
      return DictIterator<Key, Value, const_iterator_> { const_iterator_ { entryRef_.iterator_ } };
  }

private:
  explicit DictIterator(Iterator iterator): entryRef_(std::move(iterator)) {}

  DictEntryRef<Key, Value, Iterator> entryRef_;

  friend class DictIterator<Key, Value, typename detail::dict_map_type::iterator>;
  friend class Dict<Key, Value>;
  friend bool operator==<Key, Value, Iterator>(const DictIterator& lhs, const DictIterator& rhs);
};

template<class Key, class Value, class Iterator>
inline bool operator==(const DictIterator<Key, Value, Iterator>& lhs, const DictIterator<Key, Value, Iterator>& rhs) {
  return lhs.entryRef_.iterator_ == rhs.entryRef_.iterator_;
}

template<class Key, class Value, class Iterator>
inline bool operator!=(const DictIterator<Key, Value, Iterator>& lhs, const DictIterator<Key, Value, Iterator>& rhs) {
  return !(lhs == rhs);
}

}

/**
 * An object of this class stores a map from Key to Value.
 *
 * We use this class in the PyTorch kernel API instead of
 * std::unordered_map<Key, Value>, because that allows us
 * to do optimizations and switch out the underlying map
 * implementation without breaking backwards compatibility
 * for the kernel API.
 *
 * The API of this class is borrowed from std::unordered_map,
 * but with slight differences and it intentionally does not
 * support the full std::unordered_map API, because a more
 * narrow abstraction gives us more freedom to change the internals.
 */
template<class Key, class Value>
class Dict final {
private:
  // map_ stores the underlying map as a ska::flat_hash_map.
  // We intentionally don't offer conversion from/to
  // ska::flat_hash_map, return references to it or something like that,
  // because such operations would get expensive if we switch out
  // the actual map implementation.
  detail::dict_map_type map_;

public:
  using key_type = Key;
  using mapped_type = Value;
  using size_type = typename detail::dict_map_type::size_type;
  using iterator = impl::DictIterator<Key, Value, typename detail::dict_map_type::iterator>;
  using const_iterator = impl::DictIterator<Key, Value, typename detail::dict_map_type::const_iterator>;

  /**
   * Creates an empty dict.
   */
  explicit Dict() = default;

  ~Dict() = default;

  Dict(const Dict&) = default;
  Dict(Dict&&) noexcept = default;
  Dict& operator=(const Dict&) = default;
  Dict& operator=(Dict&&) noexcept = default;

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  iterator begin() {
    return iterator{map_.begin()};
  }

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  const_iterator begin() const {
    return const_iterator{map_.begin()};
  }

  /**
   * Returns an iterator to the first element of the container.
   * If the container is empty, the returned iterator will be equal to end().
   */
  const_iterator cbegin() const {
    return const_iterator{map_.cbegin()};
  }

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  iterator end() {
    return iterator{map_.end()};
  }

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  const_iterator end() const {
    return const_iterator{map_.end()};
  }

  /**
   * Returns an iterator to the element following the last element of the container.
   * This element acts as a placeholder; attempting to access it results in undefined behavior.
   */
  const_iterator cend() const {
    return const_iterator{map_.cend()};
  }

  /**
   * Checks if the container has no elements.
   */
  bool empty() const {
    return map_.empty();
  }

  /**
   * Returns the number of elements in the container.
   */
  size_type size() const {
    return map_.size();
  }

  /**
   * Erases all elements from the container. After this call, size() returns zero.
   * Invalidates any references, pointers, or iterators referring to contained elements. May also invalidate past-the-end iterators.
   */
  void clear() {
    map_.clear();
  }

  /**
   * Inserts element(s) into the container, if the container doesn't already contain an element with an equivalent key.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return A pair consisting of an iterator to the inserted element (or to the element that prevented the insertion) and a bool denoting whether the insertion took place.
   */
  template<class Key_, class Value_>
  std::pair<iterator, bool> insert(Key_&& key, Value_&& value) {
    static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert");
    static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert");
    auto inserted = map_.insert({
      Key(std::forward<Key_>(key)),
      Value(std::forward<Value_>(value))});
    return {iterator{inserted.first}, inserted.second};
  }

  /**
   * If an element with the given key already exists, it is overwritten with the given value.
   * Otherwise, a new element with the given key and value are inserted.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return The bool component is true if the insertion took place and false if the assignment took place. The iterator component is pointing at the element that was inserted or updated.
   */
  template<class Key_, class Value_>
  std::pair<iterator, bool> insert_or_assign(Key_&& key, Value_&& value) {
    static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert_or_assign");
    static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert_or_assign");
    auto inserted = map_.insert_or_assign(
      Key(std::forward<Key_>(key)),
      Value(std::forward<Value_>(value)));
    return {iterator{inserted.first}, inserted.second};
  }

  /**
   * Removes the element pointed to by iter.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   * The iterator iter must be valid and dereferenceable. Thus the end() iterator (which is valid, but is not dereferenceable) cannot be used as a value for iter.
   */
  void erase(const_iterator iter) {
    map_.erase(iter.entryRef_.iterator_);
  }

  /**
   * Removes the element with the given key, if it exists.
   * May invalidate any references, pointers, or iterators referring to contained elements.
   *
   * @return The number of elements removed. This is either '1' if an element with the key existed, or '0' if it didn't.
   */
  C10_NODISCARD size_t erase(const Key& key) {
    return map_.erase(key);
  }

  /**
   * Returns the mapped value of the element with key equivalent to key.
   * If no such element exists, an exception of type std::out_of_range is thrown.
   */
  Value at(const Key& key) {
    return map_.at(key).template to<Value>();
  }

  /**
   * Finds an element with key equivalent to key.
   *
   * @return Iterator to an element with key equivalent to key.
   *         If no such element is found, past-the-end (see end()) iterator is returned.
   */
  iterator find(const Key& key) {
    return iterator{map_.find(key)};
  }

  /**
   * Finds an element with key equivalent to key.
   *
   * @return Iterator to an element with key equivalent to key.
   *         If no such element is found, past-the-end (see end()) iterator is returned.
   */
  const_iterator find(const Key& key) const {
    return const_iterator{map_.find(key)};
  }

  /**
   * Checks if there is an element with key equivalent to key in the container.
   *
   * @return true if there is such an element, otherwise false.
   */
  bool contains(const Key& key) const {
    return end() != find(key);
  }

  /**
   * Increase the capacity so that at least count elements can be stored without
   * having to reallocate or rehash.
   */
  void reserve(size_type count) {
    map_.reserve(count);
  }
};

namespace impl {
// GenericDict is how IValue stores dicts. It is, however, not part of the
// public API. Kernels should use Dicts with concrete Key, Value types instead
// (maybe except for some internal prim ops).
using GenericDict = Dict<IValue, IValue>;
}

}

#include <ATen/core/ivalue.h>
