#pragma once

#include <c10/macros/Macros.h>

namespace at {

struct CAFFE2_API VersionUpdateMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

}
