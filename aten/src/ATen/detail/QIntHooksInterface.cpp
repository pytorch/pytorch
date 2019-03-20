#include <ATen/detail/QIntHooksInterface.h>

namespace at {

 namespace detail {
const QIntHooksInterface& getQIntHooks() {
  static std::unique_ptr<QIntHooksInterface> qint_hooks;
  // NB: The once_flag here implies that if you try to call any QInt
  // functionality before you load the qint library, you're toast.
  // Same restriction as in getCUDAHooks()
  static std::once_flag once;
  std::call_once(once, [] {
    qint_hooks = QIntHooksRegistry()->Create("QIntHooks", QIntHooksArgs{});
    if (!qint_hooks) {
      qint_hooks =
          std::unique_ptr<QIntHooksInterface>(new QIntHooksInterface());
    }
  });
  return *qint_hooks;
}
} // namespace detail

 C10_DEFINE_REGISTRY(
    QIntHooksRegistry,
    QIntHooksInterface,
    QIntHooksArgs)
}
