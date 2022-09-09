#pragma once

#include <c10/macros/Macros.h>
#include <cstddef>
#include <functional>
#include <utility>

namespace c10 {

/**
 * This template simplifies generation of simple classes that wrap an id
 * in a typesafe way. Namely, you can use it to create a very lightweight
 * type that only offers equality comparators and hashing. Example:
 *
 *   struct MyIdType final : IdWrapper<MyIdType, uint32_t> {
 *     constexpr explicit MyIdType(uint32_t id): IdWrapper(id) {}
 *   };
 *
 * Then in the global top level namespace:
 *
 *   C10_DEFINE_HASH_FOR_IDWRAPPER(MyIdType);
 *
 * That's it - equality operators and hash functions are automatically defined
 * for you, given the underlying type supports it.
 */
template <class ConcreteType, class UnderlyingType>
class IdWrapper {
 public:
  using underlying_type = UnderlyingType;
  using concrete_type = ConcreteType;

 protected:
  constexpr explicit IdWrapper(underlying_type id) noexcept(
      noexcept(underlying_type(std::declval<underlying_type>())))
      : id_(id) {}

  constexpr underlying_type underlyingId() const
      noexcept(noexcept(underlying_type(std::declval<underlying_type>()))) {
    return id_;
  }

 private:
  friend size_t hash_value(const concrete_type& v) {
    return std::hash<underlying_type>()(v.id_);
  }

  // TODO Making operator== noexcept if underlying type is noexcept equality
  // comparable doesn't work with GCC 4.8.
  //      Fix this once we don't need GCC 4.8 anymore.
  friend constexpr bool operator==(
      const concrete_type& lhs,
      const concrete_type& rhs) noexcept {
    return lhs.id_ == rhs.id_;
  }

  // TODO Making operator!= noexcept if operator== is noexcept doesn't work with
  // GCC 4.8.
  //      Fix this once we don't need GCC 4.8 anymore.
  friend constexpr bool operator!=(
      const concrete_type& lhs,
      const concrete_type& rhs) noexcept {
    return !(lhs == rhs);
  }

  underlying_type id_;
};

} // namespace c10

#define C10_DEFINE_HASH_FOR_IDWRAPPER(ClassName) \
  namespace std {                                \
  template <>                                    \
  struct hash<ClassName> {                       \
    size_t operator()(ClassName x) const {       \
      return hash_value(x);                      \
    }                                            \
  };                                             \
  }
