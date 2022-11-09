#include <c10/core/impl/CopyOnWriteContext.h>

namespace c10 {
namespace impl {

void deleteCopyOnWriteContext(void* p) {
  auto* ctx = static_cast<CopyOnWriteContext*>(p);
  ctx->decref();
}

} // namespace impl
} // namespace c10
