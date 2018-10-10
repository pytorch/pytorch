#pragma once

// Using CAFFE2_API is crucial as otherwise you'll see
// linking errors using MSVC
// See https://msdn.microsoft.com/en-us/library/a90k134d.aspx
// This header adds this if using CAFFE2_API
#include "ATen/core/ATenGeneral.h"

namespace at {
CAFFE2_API void set_num_threads(int);
CAFFE2_API int get_num_threads();
}
