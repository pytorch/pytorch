#include <ATen/detail/VULKANGuardImpl.h>

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(VULKAN, VULKANGuardImpl);

}
} // namespace at
