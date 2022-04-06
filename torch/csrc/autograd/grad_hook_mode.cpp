#include <torch/csrc/autograd/grad_hook_mode.h>

#include <c10/util/Exception.h>

namespace torch { namespace autograd {

bool &no_grad_hooks_mode() {
    static bool no_grad_hooks_mode;
    return no_grad_hooks_mode;
}

void NoGradHooksMode::enter() {
  if (no_grad_hooks_mode()) {
    TORCH_CHECK(
        false,
        "no grad hooks mode has already been set. We do not yet support nested no grad hooks ",
        "mode.")
  }
  no_grad_hooks_mode() = true;
}

void NoGradHooksMode::exit() {
  TORCH_INTERNAL_ASSERT(no_grad_hooks_mode(), "exiting no grad hooks mode but it wasn't set!");
  no_grad_hooks_mode() = false;
}

}}
