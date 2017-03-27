#pragma once

#include "Traits.hpp"
#include <THC/THCHalf.h>

namespace thpp {

template<>
struct type_traits<half> {
  static constexpr Type type = Type::HALF;
  static constexpr bool is_floating_point = true;
};

} // namespace thpp
