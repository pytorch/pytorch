#include <ATen/detail/UniqueVoidPtr.h>

namespace at { namespace detail {

void deleteNothing(void*) {}

}} // namespace at
