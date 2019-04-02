#pragma once

#include <c10/macros/Macros.h>

namespace at {

// This thread-local mode controls whether in-place operations on a Tensor
// update its version.
//
// WARNING: You should not disable version update and perform in-place operations
// on a Tensor that is a saved variable in an autograd graph, because that could
// result in incorrect gradient calculations.
struct CAFFE2_API VersionUpdateMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

}
