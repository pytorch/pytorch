#pragma once

#include <c10/Registry.h>

// Kept for BC reasons
#include "caffe2/core/common.h"
#include "caffe2/core/typeid.h"

namespace caffe2 {
  using ::c10::Registry;
}

#define CAFFE_ANONYMOUS_VARIABLE(name) C10_ANONYMOUS_VARIABLE(name)
