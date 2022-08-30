#pragma once

#include <ATen/native/special_functions/detail/promote.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename... T>
using promote_t = typename promote<T...>::type;
}
}
}
}
