// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#import "MetalOpTestRunner.h"

#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>
#import <ATen/native/metal/mpscnn/tests/MetalOpTestRunner.h>

@implementation MetalOpTestRunner {
  NSMutableDictionary* _tests;
}

+ (instancetype)sharedInstance {
  static dispatch_once_t onceToken;
  static MetalOpTestRunner* instance = nil;
  dispatch_once(&onceToken, ^{
    instance = [MetalOpTestRunner new];
  });
  return instance;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    [self registerTests];
  }
  return self;
}

- (void)registerTests {
  _tests = [NSMutableDictionary dictionary];
#define REG_TEST(arg1, arg2)    \
  _tests[@arg1] = ^BOOL(void) { \
    return arg2();              \
  }
  REG_TEST("test_synchronization", test_synchronization);
  REG_TEST("test_copy_nchw_to_metal", test_copy_nchw_to_metal);
  REG_TEST("test_conv2d", test_conv2d);
  REG_TEST("test_depthwiseConv", test_depthwiseConv);
  REG_TEST("test_max_pool2d", test_max_pool2d);
  REG_TEST("test_max_pool2d_ceil", test_max_pool2d_ceil);
  REG_TEST("test_relu", test_relu);
  REG_TEST("test_addmm", test_addmm);
  REG_TEST("test_add", test_add);
  REG_TEST("test_add_broadcast", test_add_broadcast);
  REG_TEST("test_add_broadcast2", test_add_broadcast2);
  REG_TEST("test_sub", test_sub);
  REG_TEST("test_sub_broadcast", test_sub_broadcast);
  REG_TEST("test_mul", test_mul);
  REG_TEST("test_mul_broadcast", test_mul_broadcast);
  REG_TEST("test_mul_broadcast2", test_mul_broadcast2);
  REG_TEST("test_div", test_div);
  REG_TEST("test_div_broadcast", test_div_broadcast);
  REG_TEST("test_div_broadcast2", test_div_broadcast2);
  REG_TEST("test_t", test_t);
  REG_TEST("test_transpose", test_transpose);
  REG_TEST("test_transpose2", test_transpose2);
  REG_TEST("test_transpose3", test_transpose3);
  REG_TEST("test_view", test_view);
  REG_TEST("test_view2", test_view2);
  REG_TEST("test_view3", test_view3);
  REG_TEST("test_view4", test_view4);
  REG_TEST("test_cat_dim0", test_cat_dim0);
  REG_TEST("test_cat_dim0_nonarray", test_cat_dim0_nonarray);
  REG_TEST("test_cat_dim1_0", test_cat_dim1_0);
  REG_TEST("test_cat_dim1_1", test_cat_dim1_1);
  REG_TEST("test_cat_dim1_nonarray_0", test_cat_dim1_nonarray_0);
  REG_TEST("test_cat_dim1_nonarray_1", test_cat_dim1_nonarray_1);
  REG_TEST("test_softmax", test_softmax);
  REG_TEST("test_sigmoid", test_sigmoid);
  REG_TEST("test_hardsigmoid", test_hardsigmoid);
  REG_TEST("test_hardswish_", test_hardswish_);
  REG_TEST("test_hardswish", test_hardswish);
  REG_TEST("test_hardshrink_", test_hardshrink_);
  REG_TEST("test_hardshrink", test_hardshrink);
  REG_TEST("test_leaky_relu_", test_leaky_relu_);
  REG_TEST("test_leaky_relu", test_leaky_relu);
  REG_TEST("test_upsampling_nearest2d_vec", test_upsampling_nearest2d_vec);
  REG_TEST("test_upsampling_nearest2d_vec2", test_upsampling_nearest2d_vec2);
  REG_TEST("test_adaptive_avg_pool2d", test_adaptive_avg_pool2d);
  REG_TEST("test_hardtanh_", test_hardtanh_);
  REG_TEST("test_hardtanh", test_hardtanh);
  REG_TEST("test_reshape", test_reshape);
  REG_TEST("test_chunk", test_chunk);
  REG_TEST("test_chunk3", test_chunk3);
  REG_TEST("test_reflection_pad2d", test_reflection_pad2d);
#if !TARGET_IPHONE_SIMULATOR
  REG_TEST("test_mean_dim", test_mean_dim);
  REG_TEST("test_mean_dim2", test_mean_dim2);
  REG_TEST("test_mean_dim3", test_mean_dim3);
  REG_TEST("test_chunk2", test_chunk2);
#endif
}

- (NSDictionary*)tests {
  return _tests;
}

@end
