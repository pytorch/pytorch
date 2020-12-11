#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED() && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
// Coming soon
#endif  // AT_CUDNN_ENABLED and CUDNN_VERSION
