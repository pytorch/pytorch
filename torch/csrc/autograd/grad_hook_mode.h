#pragma once

#include <c10/macros/Export.h>

namespace torch { namespace autograd {

bool &no_grad_hooks_mode();

struct TORCH_API NoGradHooksMode {
  // Enter no grad hooks mode, causing AccumulateGrad to not fire gradient
  // hooks when invoked.
  static void enter();

  // Exit no grad hooks mode
  static void exit();
};

}}
