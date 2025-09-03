#include <ATen/ATen.h>

#ifdef AT_CUDNN_ENABLED
#error "AT_CUDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKL_ENABLED
#error "AT_MKL_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKLDNN_ENABLED
#error "AT_MKLDNN_ENABLED should not be visible in public headers"
#endif

#ifdef AT_MKLDNN_ACL_ENABLED
#error "AT_MKLDNN_ACL_ENABLED should not be visible in public headers"
#endif

#ifdef CAFFE2_STATIC_LINK_CUDA
#error "CAFFE2_STATIC_LINK_CUDA should not be visible in public headers"
#endif

#include <gtest/gtest.h>

TEST(VerifyApiVisibility, Test) {
  ASSERT_EQ(1, 1);
}
