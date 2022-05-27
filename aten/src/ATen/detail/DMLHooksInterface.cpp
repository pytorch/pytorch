#include <ATen/detail/DMLHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const DMLHooksInterface& getDMLHooks() {
  static std::unique_ptr<DMLHooksInterface> dml_hooks;
  static std::once_flag once;
  std::call_once(once, [] {
    dml_hooks = DMLHooksRegistry()->Create("DMLHooks", {});
    if (!dml_hooks) {
      dml_hooks =
          // NOLINTNEXTLINE(modernize-make-unique)
          std::unique_ptr<DMLHooksInterface>(new DMLHooksInterface());
    }
  });
  return *dml_hooks;
}
} // namespace detail

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(DMLHooksRegistry, DMLHooksInterface, DMLHooksArgs)

} // namespace at
