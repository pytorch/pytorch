#pragma once

#include <cudnn.h>

#define STRINGIFY(x) #x
#define STRING(x) STRINGIFY(x)

#if CUDNN_MAJOR < 6
#pragma message ("CuDNN v" STRING(CUDNN_MAJOR) " found, but need at least CUDNN v6. You can get the latest version of CUDNN from https://developer.nvidia.com/cudnn")
#error "Old version of CuDNN found but need at least CuDNN v6. You can get the latest version of CuDNN from https://developer.nvidia.com/cudnn"
#endif

#undef STRINGIFY
#undef STRING

