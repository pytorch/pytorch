#include <ATen/core/UniqueVoidPtr.h>

namespace at {
namespace detail {

void deleteNothing(void*) {}

} // namespace detail
} // namespace at
