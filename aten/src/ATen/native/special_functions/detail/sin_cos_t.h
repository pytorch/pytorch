#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct sin_cos_t {
  T1 sin_v;
  T1 cos_v;
};
}
}
}
}
