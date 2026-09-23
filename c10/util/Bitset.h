#pragma once

#include <bit>
#include <bitset>
#include <cstddef>

namespace c10::utils {

using bitset = std::bitset<64>;

template <class Func>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
void for_each_set_bit(bitset b, Func&& func) {
  for (auto val = b.to_ullong(); val; val = val & (val - 1)) {
    func(static_cast<size_t>(std::countr_zero(val)));
  }
}

} // namespace c10::utils
