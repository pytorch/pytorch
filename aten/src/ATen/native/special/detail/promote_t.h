#pragma once

#include <ATen/native/special/detail/promote.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename... T>
using promote_t = typename promote<T...>::type;
}
}
}
}
