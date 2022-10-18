#include <ATen/core/dispatch/Dispatcher.h>

namespace at {
namespace impl {

class TORCH_API PythonOpRegistrationTrampoline final : public c10::OperatorKernel {
  // This DispatchKey tells us what dispatch key the Python operator was
  // registered from.  This is DIFFERENT from the highestPriorityId from
  // the passed DispatchKeySet; you might get CompositeExplicitAutograd for
  // the former, but CPU for the latter.  It is easiest to just remember
  // what the key this trampoline is for at registration time.
  c10::DispatchKey dispatch_key_;

  static std::atomic<c10::impl::PyInterpreter*> interpreter_;

public:

  static void registerInterpreter(c10::impl::PyInterpreter*);

  PythonOpRegistrationTrampoline(c10::DispatchKey dispatch_key) : dispatch_key_(dispatch_key) {}
  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack);

};

} // namespace impl
} // namespace at
