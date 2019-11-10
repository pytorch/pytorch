#pragma once

#include <c10/macros/Macros.h>

namespace at {
namespace autocast {

struct CAFFE2_API AutocastMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
  static void clear_cache();
};

}
}
