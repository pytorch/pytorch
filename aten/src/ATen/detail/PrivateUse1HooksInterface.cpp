#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at {
namespace detail {

const PrivateUse1HooksInterface& getPrivateUse1Hooks() {
  static PrivateUse1HooksInterface* privateuse1_hooks = []() {
    auto hooks =
        PrivateUse1HooksRegistry()->Create("PrivateUse1Hooks").release();
    if (!hooks) {
      hooks = new PrivateUse1HooksInterface();
    }

    return hooks;
  }();

  return *privateuse1_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, PrivateUse1HooksInterface)

} // namespace at
