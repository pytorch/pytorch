#pragma once

#include <cudnn.h>

#define STRINGIFY(x) #x
#define STRING(x) STRINGIFY(x)

#if CUDNN_MAJOR < 6
#pragma message ("CuDNN v" STRING(CUDNN_MAJOR) " found, but need at least CuDNN v6. You can get the latest version of CuDNN from https://developer.nvidia.com/cudnn or disable CuDNN with USE_CUDNN=0")
#pragma message "We strongly encourage you to move to 6.0 and above."
#pragma message "This message is intended to annoy you enough to update."
#endif

#undef STRINGIFY
#undef STRING

