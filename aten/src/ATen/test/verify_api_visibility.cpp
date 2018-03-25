#include <ATen/ATen.h>

#ifdef AT_CUDA_ENABLED
#error "AT_CUDA_ENABLED should not be visible in public headers"
#endif

#ifdef AT_CUDNN_ENABLED
#error "AT_CUDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKL_ENABLED
#error "AT_MKL_ENABLED should not be visible in public headers"
#endif

auto main() -> int {}
