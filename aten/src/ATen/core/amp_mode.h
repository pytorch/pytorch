#pragma once

#include <c10/macros/Macros.h>

namespace at {

// C++ API should mirror Python API.
// Dunno why CAFFE2_API needs to be here, cargo culting grad_mode.h
struct CAFFE2_API AmpMode {
  static bool is_grad_scaling_enabled();
  static void set_grad_scaling_enabled(bool new_enabled);
  static float get_grad_scale();
  static void set_grad_scale(float new_scale);
};

}
