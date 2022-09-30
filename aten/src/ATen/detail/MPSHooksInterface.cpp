#include <ATen/detail/MPSHooksInterface.h>
#include <c10/util/Exception.h>

namespace at {
namespace detail {

const MPSHooksInterface& getMPSHooks() {
  static std::unique_ptr<MPSHooksInterface> mps_hooks;
#if !defined C10_MOBILE
  static std::once_flag once;
  std::call_once(once, [] {
    mps_hooks = MPSHooksRegistry()->Create("MPSHooks", MPSHooksArgs{});
    if (!mps_hooks) {
      mps_hooks =
          // NOLINTNEXTLINE(modernize-make-unique)
          std::unique_ptr<MPSHooksInterface>(new MPSHooksInterface());
    }
  });
#else
  if (mps_hooks == nullptr) {
    mps_hooks =
        // NOLINTNEXTLINE(modernize-make-unique)
        std::unique_ptr<MPSHooksInterface>(new MPSHooksInterface());
  }
#endif
  return *mps_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs)

} // namespace at
