#include <ATen/ATen.h>

#ifdef AT_CUDA_ENABLED
#error "AT_CUDA_ENABLED should not be visible in public headers"
#endif

auto main() -> int {}
