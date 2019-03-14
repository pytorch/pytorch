#include <ATen/detail/XLAHooksInterface.h>

#include <atomic>

namespace at {
namespace detail {
namespace {

std::atomic<XLAHooksInterface*> g_xla_hooks(new XLAHooksInterface());

}  // namespace

const XLAHooksInterface& getXLAHooks() {
  return *g_xla_hooks.load();
}

void setXLAHooks(XLAHooksInterface* xla_hooks) {
  g_xla_hooks = xla_hooks;
}

} // namespace detail
} // namespace at
