#include <ATen/detail/CPUGuardImpl.h>

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(CPU, CPUGuardImpl);

}} // namespace at::detail
