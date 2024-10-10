#include <ATen/ATen.h>

#ifdef AT_CUDNN_ENABLED
#error "AT_CUDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKL_ENABLED
#error "AT_MKL_ENABLED should not be visible in public headers"
#endif

#ifdef AT_ONEDNN_ENABLED
#error "AT_ONEDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_ONEDNN_ACL_ENABLED
#error "AT_ONEDNN_ACL_ENABLED should not be visible in public headers"
#endif

#ifdef CAFFE2_STATIC_LINK_CUDA
#error "CAFFE2_STATIC_LINK_CUDA should not be visible in public headers"
#endif

auto main() -> int {}
