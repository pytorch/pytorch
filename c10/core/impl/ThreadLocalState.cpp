#include <c10/core/impl/ThreadLocalState.h>

namespace c10 {
namespace impl {

thread_local PODLocalState raw_thread_local_state;

#if defined(_MSC_VER) || defined(C10_ANDROID)
PODLocalState* _get_thread_local_state() {
  return &raw_thread_local_state;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID)

} // namespace impl
} // namespace c10
