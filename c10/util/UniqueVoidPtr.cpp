#include <c10/util/UniqueVoidPtr.h>

namespace c10 {
namespace detail {

void deleteNothing(void*) {}

} // namespace detail
} // namespace c10
