#ifndef MPSCNNTests_h
#define MPSCNNTests_h

namespace at {
namespace native {
namespace metal {

bool test_aten();
bool test_NC4();
bool test_MPSImage();
bool test_MPSImageCopy();
bool test_MPSTemporaryImageCopy();
bool test_conv2d();
bool test_depthwiseConv();
bool test_max_pool2d();
bool test_relu();
bool test_addmm();
bool test_add();
bool test_sub();
bool test_mul();
bool test_t();
bool test_view();
bool test_softmax();
bool test_sigmoid();
bool test_upsampling_nearest2d_vec();
bool test_adaptive_avg_pool2d();
bool test_hardtanh_();
bool test_reshape();

} // namespace metal
} // namespace native
} // namespace at

#endif
