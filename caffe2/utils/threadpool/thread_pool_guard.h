#pragma once

#include <c10/macros/Macros.h>

namespace caffe2 {

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API _NoPThreadPoolGuard {
  static bool is_enabled();
  static void set_enabled(bool enabled);

  _NoPThreadPoolGuard(): prev_mode_(_NoPThreadPoolGuard::is_enabled()) {
      _NoPThreadPoolGuard::set_enabled(true);
  }
  ~_NoPThreadPoolGuard() {
      _NoPThreadPoolGuard::set_enabled(prev_mode_);
  }
  private:
    bool prev_mode_;
};

}
