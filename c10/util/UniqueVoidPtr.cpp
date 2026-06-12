#include <c10/util/UniqueVoidPtr.h>

namespace c10::detail {

void deleteNothing(void* /*unused*/) {}

} // namespace c10::detail
