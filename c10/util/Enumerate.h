/*
 * Ported from folly/container/Enumerate.h
 */

#pragma once

#include <iterator>
#include <memory>

#ifdef _WIN32
#include <basetsd.h> // @manual
using ssize_t = SSIZE_T;
#endif

#include <c10/macros/Macros.h>

/**
 * Similar to Python's enumerate(), enumerate() can be used to
 * iterate a range with a for-range loop, and it also allows to
 * retrieve the count of iterations so far. Can be used in constexpr
 * context.
 *
 * For example:
 *
 * for (auto&& [index, element] : enumerate(vec)) {
 *   // index is a const reference to a size_t containing the iteration count.
 *   // element is a reference to the type contained within vec, mutable
 *   // unless vec is const.
 * }
 *
 * If the binding is const, the element reference is too.
 *
 * for (const auto&& [index, element] : enumerate(vec)) {
 *   // element is always a const reference.
 * }
 *
 * It can also be used as follows:
 *
 * for (auto&& it : enumerate(vec)) {
 *   // *it is a reference to the current element. Mutable unless vec is const.
 *   // it->member can be used as well.
 *   // it.index contains the iteration count.
 * }
 *
 * As before, const auto&& it can also be used.
 */

namespace c10 {

namespace detail {

template <class T>
struct MakeConst {
  using type = const T;
};
template <class T>
struct MakeConst<T&> {
  using type = const T&;
};
template <class T>
struct MakeConst<T*> {
  using type = const T*;
};

template <class Iterator>
class Enumerator {
 public:
  constexpr explicit Enumerator(Iterator it) : it_(std::move(it)) {}

  class Proxy {
   public:
    using difference_type = ssize_t;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using iterator_category = std::input_iterator_tag;

    C10_ALWAYS_INLINE constexpr explicit Proxy(const Enumerator& e)
        : index(e.idx_), element(*e.it_) {}

    // Non-const Proxy: Forward constness from Iterator.
    C10_ALWAYS_INLINE constexpr reference operator*() {
      return element;
    }
    C10_ALWAYS_INLINE constexpr pointer operator->() {
      return std::addressof(element);
    }

    // Const Proxy: Force const references.
    C10_ALWAYS_INLINE constexpr typename MakeConst<reference>::type operator*()
        const {
      return element;
    }
    C10_ALWAYS_INLINE constexpr typename MakeConst<pointer>::type operator->()
        const {
      return std::addressof(element);
    }

   public:
    size_t index;
    reference element;
  };

  C10_ALWAYS_INLINE constexpr Proxy operator*() const {
    return Proxy(*this);
  }

  C10_ALWAYS_INLINE constexpr Enumerator& operator++() {
    ++it_;
    ++idx_;
    return *this;
  }

  template <typename OtherIterator>
  C10_ALWAYS_INLINE constexpr bool operator==(
      const Enumerator<OtherIterator>& rhs) const {
    return it_ == rhs.it_;
  }

  template <typename OtherIterator>
  C10_ALWAYS_INLINE constexpr bool operator!=(
      const Enumerator<OtherIterator>& rhs) const {
    return !(it_ == rhs.it_);
  }

 private:
  template <typename OtherIterator>
  friend class Enumerator;

  Iterator it_;
  size_t idx_ = 0;
};

template <class Range>
class RangeEnumerator {
  Range r_;
  using BeginIteratorType = decltype(std::declval<Range>().begin());
  using EndIteratorType = decltype(std::declval<Range>().end());

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  constexpr explicit RangeEnumerator(Range&& r) : r_(std::forward<Range>(r)) {}

  constexpr Enumerator<BeginIteratorType> begin() {
    return Enumerator<BeginIteratorType>(r_.begin());
  }
  constexpr Enumerator<EndIteratorType> end() {
    return Enumerator<EndIteratorType>(r_.end());
  }
};

} // namespace detail

template <class Range>
constexpr detail::RangeEnumerator<Range> enumerate(Range&& r) {
  return detail::RangeEnumerator<Range>(std::forward<Range>(r));
}

} // namespace c10
