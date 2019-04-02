#pragma once

#include <c10/macros/Macros.h>

namespace at {

// This thread-local mode controls whether in-place operations on a Tensor
// update its version.
//
// WARNING: Disabling version update and performing in-place operations on a
// Tensor that is a saved variable in an autograd graph could result in
// incorrect gradient calculations. You should make sure that no incorrect
// gradient calculations could occur in your code, before disabling version update.
struct CAFFE2_API VersionUpdateMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

}
