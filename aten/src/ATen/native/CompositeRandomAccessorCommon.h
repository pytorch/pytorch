#include <utility>

#pragma once

namespace at { namespace native {

namespace {

// operator_brackets_proxy is used in
// CompositeRandomAccessor in place of operator[].
// For some iterators, references returned by operator[]
// could become invalid, operator_brackets_proxy tries to
// resolve that by making accessor[n] to be equivalent to
// *(accessor + n).
template <typename Accessor>
class operator_brackets_proxy {
  using reference = typename std::iterator_traits<Accessor>::reference;
  using value_type = typename std::iterator_traits<Accessor>::value_type;

public:
  C10_HOST_DEVICE
  operator_brackets_proxy(Accessor const& accessor)
    : accessor(accessor)
  {}

  C10_HOST_DEVICE
  operator reference() {
    return *accessor;
  }

  C10_HOST_DEVICE
  reference operator*() {
    return *accessor;
  }

  C10_HOST_DEVICE
  operator_brackets_proxy& operator=(value_type const& val) {
    *accessor = val;
    return *this;
  }

private:
  Accessor accessor;
};

}

// references_holder is used as a surrogate for the
// references type from std::iterator_traits in CompositeRandomAccessor.
// It is assumed in CompositeRandomAccessor that
// References = tuple<Types&...>,
// Values = tuple<Types...> by default,
// but they could be anything as long as References could be
// cast to Values.
// If you plan to use it with STL, for example, you will need to
// define 'swap` and `get`(aka std::get) methods.
template <typename Values, typename References>
class references_holder {
public:
  using values = Values;
  using references = References;

  C10_HOST_DEVICE
  references_holder(references refs)
    : refs{std::move(refs)}
  {}

  C10_HOST_DEVICE
  operator references() {
    return refs;
  }

  C10_HOST_DEVICE
  operator values() {
    return refs;
  }

  C10_HOST_DEVICE
  references_holder& operator=(values vals) {
    refs = vals;
    return *this;
  }

  C10_HOST_DEVICE
  references& data() {
    return refs;
  }

protected:
  references refs;
};

// CompositeRandomAccessor is essentially a simplified version of
// a random access iterator over two random access iterators.
// TupleInfo should contain a variadic type `tuple`, and a method `tie`,
// which constructs a tuple of references from a variadic list of arguments.
template <typename KeyAccessor, typename ValueAccessor, typename TupleInfo>
class CompositeRandomAccessor {
  using self_type = CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfo>;

  using key_accessor_value_type =
    typename std::iterator_traits<KeyAccessor>::value_type;
  using value_accessor_value_type =
    typename std::iterator_traits<ValueAccessor>::value_type;
  using key_accessor_reference_type =
    typename std::iterator_traits<KeyAccessor>::reference;
  using value_accessor_reference_type =
    typename std::iterator_traits<ValueAccessor>::reference;

  using composite_value_type = typename TupleInfo::template tuple<
    key_accessor_value_type,
    value_accessor_value_type>;
  using composite_reference = typename TupleInfo::template tuple<
    key_accessor_reference_type,
    value_accessor_reference_type>;

public:
  using value_type = composite_value_type;
  using reference = references_holder<composite_value_type, composite_reference>;
  // Note that CompositeRandomAccessor does not hold key and values
  // in a specific datastrcture, which means that a pointer to a (key, value)
  // is not defined. Hence we just use a pointer type of the KeyAccessor.
  using pointer = typename std::iterator_traits<KeyAccessor>::pointer;
  using difference_type = typename std::iterator_traits<KeyAccessor>::difference_type;
  using iterator_category = std::random_access_iterator_tag;

  C10_HOST_DEVICE
  CompositeRandomAccessor() = default;

  C10_HOST_DEVICE
  CompositeRandomAccessor(KeyAccessor keys, ValueAccessor values)
    : keys(keys), values(values)
  {}

  // Pointer-like operations {
  C10_HOST_DEVICE
  reference operator*() const {
    return TupleInfo::tie(*keys, *values);
  }

  // operator->() is supposed to return a pointer type.
  // Since CompositeRandomAccessor does not hold pointers to pairs,
  // we just return a pointer to a key.
  C10_HOST_DEVICE
  auto* operator->() const {
    return keys.operator->();
  }

  C10_HOST_DEVICE
  reference operator[](difference_type idx) {
    return operator_brackets_proxy<self_type>(
      CompositeRandomAccessor(keys + idx, values + idx)
    );
  }
  // }

  // Prefix/postfix increment/decrement {
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator++() {
    ++keys;
    ++values;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator++(int) {
    CompositeRandomAccessor copy(*this);
    ++*this;
    return copy;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor& operator--() {
    --keys;
    --values;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator--(int) {
    CompositeRandomAccessor copy(*this);
    --*this;
    return copy;
  }
  // }

  // Arithmetic operations {
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator+=(difference_type offset) {
    keys += offset;
    values += offset;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator+(difference_type offset) const {
    return CompositeRandomAccessor(keys + offset, values + offset);
  }

  C10_HOST_DEVICE
  friend CompositeRandomAccessor operator+(
    difference_type offset,
    const CompositeRandomAccessor& accessor
  ) {
    return accessor + offset;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor& operator-=(difference_type offset) {
    keys -= offset;
    values -= offset;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator-(difference_type offset) const {
    return CompositeRandomAccessor(keys - offset, values - offset);
  }

  C10_HOST_DEVICE
  difference_type operator-(const CompositeRandomAccessor& other) const {
    return keys - other.keys;
  }
  // }

  // Comparison operators {
  C10_HOST_DEVICE
  bool operator==(const CompositeRandomAccessor& other) const {
    return keys == other.keys;
  }

  C10_HOST_DEVICE
  bool operator!=(const CompositeRandomAccessor& other) const {
    return keys != other.keys;
  }

  C10_HOST_DEVICE
  bool operator<(const CompositeRandomAccessor& other) const {
    return keys < other.keys;
  }

  C10_HOST_DEVICE
  bool operator<=(const CompositeRandomAccessor& other) const {
    return keys <= other.keys;
  }

  C10_HOST_DEVICE
  bool operator>(const CompositeRandomAccessor& other) const {
    return keys > other.keys;
  }

  C10_HOST_DEVICE
  bool operator>=(const CompositeRandomAccessor& other) const {
    return keys >= other.keys;
  }
  // }

protected:
  KeyAccessor keys;
  ValueAccessor values;
};

}} // namespace at::native
