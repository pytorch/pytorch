#pragma once

#include <c10/macros/Export.h>

// There's no difference between aten, torch and caffe2 libs any more
// TODO: clean up the naming for consistency
#define TORCH_API CAFFE2_API

#ifdef _WIN32
#define TORCH_PYTHON_API
#else
#define TORCH_PYTHON_API CAFFE2_API
#endif
