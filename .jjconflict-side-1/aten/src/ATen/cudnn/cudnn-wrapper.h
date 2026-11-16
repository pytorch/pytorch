#pragma once

#include <cudnn.h>

#define STRINGIFY(x) #x
#define STRING(x) STRINGIFY(x)

#if CUDNN_MAJOR < 8 || (CUDNN_MAJOR == 8 && CUDNN_MINOR < 5)
#pragma message("CuDNN v" STRING( \
    CUDNN_MAJOR) " found, but need at least CuDNN v8. You can get the latest version of CuDNN from https://developer.nvidia.com/cudnn or disable CuDNN with USE_CUDNN=0")
#pragma message "We strongly encourage you to move to 8.5 and above."
#pragma message "This message is intended to annoy you enough to update."
#endif

#undef STRINGIFY
#undef STRING
