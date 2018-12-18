#include <ATen/detail/MKLDNNHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {
// See getCUDAHooks for some more commentary
const MKLDNNHooksInterface& getMKLDNNHooks() {
  static std::unique_ptr<MKLDNNHooksInterface> mkldnn_hooks;
  static std::once_flag once;
  std::call_once(once, [] {
    mkldnn_hooks = MKLDNNHooksRegistry()->Create("MKLDNNHooks", MKLDNNHooksArgs{});
    if (!mkldnn_hooks) {
      mkldnn_hooks =
          std::unique_ptr<MKLDNNHooksInterface>(new MKLDNNHooksInterface());
    }
  });
  return *mkldnn_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MKLDNNHooksRegistry, MKLDNNHooksInterface, MKLDNNHooksArgs)

} // namespace at
