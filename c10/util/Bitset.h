#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/Optional.h>
#include <iostream>

namespace c10 {
namespace utils {

/**
 * This is a simple bitset class with sizeof(long long int) bits.
 * You can set bits, unset bits, query bits by index,
 * and query for the first set bit.
 * Before using this class, please also take a look at std::bitset,
 * which has more functionality and is more generic. It is probably
 * a better fit for your use case. The sole reason for c10::utils::bitset
 * to exist is that std::bitset misses a find_first_set() method.
 */
struct bitset final {
 public:
  static constexpr size_t NUM_BITS = 8 * sizeof(long long int);

  constexpr bitset() noexcept : bitset_(0) {}
  constexpr bitset(const bitset&) noexcept = default;
  constexpr bitset(bitset&&) noexcept = default;
  constexpr bitset& operator=(const bitset&) noexcept = default;
  constexpr bitset& operator=(bitset&&) noexcept = default;

  constexpr void set(size_t index) noexcept {
    bitset_ |= (static_cast<long long int>(1) << index);
  }

  constexpr void unset(size_t index) noexcept {
    bitset_ &= ~(static_cast<long long int>(1) << index);
  }

  constexpr bool get(size_t index) const noexcept {
    return bitset_ & (static_cast<long long int>(1) << index);
  }

  // Call the given functor with the index of each bit that is set
  template <class Func>
  void for_each_set_bit(Func&& func) const {
    bitset cur = *this;
    size_t index = ffsll(cur.bitset_);
    while (0 != index) {
      // -1 because ffsll() is not zero-indices but the first bit
      // is returned as index 1.
      index -= 1;
      func(index);
      cur.unset(index);
      index = ffsll(cur.bitset_);
    }
  }

 private:
  long long int bitset_;
};

} // namespace utils
} // namespace c10
