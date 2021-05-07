// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>
#import <ATen/native/metal/op_test_runner/MetalOpTestRunner.h>

static NSArray<NSString*>* const testNames = @[
  @"test_synchronization",
  @"test_nchw_to_nc4_cpu",
  @"test_copy_nchw_to_metal",
  @"test_conv2d",
  @"test_depthwiseConv",
  @"test_max_pool2d",
  @"test_max_pool2d_ceil",
  @"test_relu",
  @"test_addmm",
  @"test_add",
  @"test_add_broadcast",
  @"test_add_broadcast2",
  @"test_sub",
  @"test_sub_broadcast",
  @"test_mul",
  @"test_mul_broadcast",
  @"test_mul_broadcast2",
  @"test_div",
  @"test_div_broadcast",
  @"test_div_broadcast2",
  @"test_t",
  @"test_transpose",
  @"test_transpose2",
  @"test_transpose3",
  @"test_view",
  @"test_view2",
  @"test_view3",
  @"test_view4",
  @"test_cat_dim0",
  @"test_cat_dim0_nonarray",
  @"test_cat_dim1_0",
  @"test_cat_dim1_1",
  @"test_cat_dim1_nonarray_0",
  @"test_cat_dim1_nonarray_1",
  @"test_softmax",
  @"test_sigmoid",
  @"test_hardsigmoid",
  @"test_hardswish",
  @"test_upsampling_nearest2d_vec",
  @"test_adaptive_avg_pool2d",
  @"test_hardtanh_",
  @"test_reshape",
  @"test_mean_dim",
  @"test_mean_dim2",
  @"test_mean_dim3"
];

@implementation MetalOpTestRunner

+ (NSDictionary<NSString*, NSNumber*>*)testMPSCNNOps {
  NSMutableDictionary<NSString*, NSNumber*>* resultDict =
      [NSMutableDictionary dictionary];
  for (NSString* testName in testNames) {
    resultDict[testName] = [MetalOpTestRunner runTestWithName:testName];
  }
  return resultDict;
}

+ (NSNumber*)runTestWithName:(NSString*)testName {
  BOOL res = NO;
  if ([testName isEqualToString:@"test_synchronization"]) {
    res = test_synchronization();
  } else if ([testName isEqualToString:@"test_nchw_to_nc4_cpu"]) {
    res = test_nchw_to_nc4_cpu();
  } else if ([testName isEqualToString:@"test_copy_nchw_to_metal"]) {
    res = test_copy_nchw_to_metal();
  } else if ([testName isEqualToString:@"test_nchw_to_nc4_cpu"]) {
    res = test_nchw_to_nc4_cpu();
  } else if ([testName isEqualToString:@"test_conv2d"]) {
    res = test_conv2d();
  } else if ([testName isEqualToString:@"test_depthwiseConv"]) {
    res = test_depthwiseConv();
  } else if ([testName isEqualToString:@"test_max_pool2d"]) {
    res = test_max_pool2d();
  } else if ([testName isEqualToString:@"test_max_pool2d_ceil"]) {
    res = test_max_pool2d_ceil();
  } else if ([testName isEqualToString:@"test_relu"]) {
    res = test_relu();
  } else if ([testName isEqualToString:@"test_addmm"]) {
    res = test_addmm();
  } else if ([testName isEqualToString:@"test_add"]) {
    res = test_add();
  } else if ([testName isEqualToString:@"test_add_broadcast"]) {
    res = test_add_broadcast();
  } else if ([testName isEqualToString:@"test_add_broadcast2"]) {
    res = test_add_broadcast2();
  } else if ([testName isEqualToString:@"test_sub"]) {
    res = test_sub();
  } else if ([testName isEqualToString:@"test_sub_broadcast"]) {
    res = test_sub_broadcast();
  } else if ([testName isEqualToString:@"test_sub_broadcast2"]) {
    res = test_sub_broadcast2();
  } else if ([testName isEqualToString:@"test_mul"]) {
    res = test_mul();
  } else if ([testName isEqualToString:@"test_mul_broadcast"]) {
    res = test_mul_broadcast();
  } else if ([testName isEqualToString:@"test_mul_broadcast2"]) {
    res = test_mul_broadcast2();
  } else if ([testName isEqualToString:@"test_div"]) {
    res = test_div();
  } else if ([testName isEqualToString:@"test_div_broadcast"]) {
    res = test_div_broadcast();
  } else if ([testName isEqualToString:@"test_div_broadcast2"]) {
    res = test_div_broadcast2();
  } else if ([testName isEqualToString:@"test_t"]) {
    res = test_t();
  } else if ([testName isEqualToString:@"test_transpose"]) {
    res = test_transpose();
  } else if ([testName isEqualToString:@"test_transpose2"]) {
    res = test_transpose2();
  } else if ([testName isEqualToString:@"test_transpose3"]) {
    res = test_transpose3();
  } else if ([testName isEqualToString:@"test_view"]) {
    res = test_view();
  } else if ([testName isEqualToString:@"test_view2"]) {
    res = test_view2();
  } else if ([testName isEqualToString:@"test_view3"]) {
    res = test_view3();
  } else if ([testName isEqualToString:@"test_view4"]) {
    res = test_view4();
  } else if ([testName isEqualToString:@"test_cat_dim0"]) {
    res = test_cat_dim0();
  } else if ([testName isEqualToString:@"test_cat_dim0_nonarray"]) {
    res = test_cat_dim0_nonarray();
  } else if ([testName isEqualToString:@"test_cat_dim1_0"]) {
    res = test_cat_dim1_0();
  } else if ([testName isEqualToString:@"test_cat_dim1_1"]) {
    res = test_cat_dim1_1();
  } else if ([testName isEqualToString:@"test_cat_dim1_nonarray_0"]) {
    res = test_cat_dim1_nonarray_0();
  } else if ([testName isEqualToString:@"test_cat_dim1_nonarray_1"]) {
    res = test_cat_dim1_nonarray_1();
  } else if ([testName isEqualToString:@"test_softmax"]) {
    res = test_softmax();
  } else if ([testName isEqualToString:@"test_sigmoid"]) {
    res = test_sigmoid();
  } else if ([testName isEqualToString:@"test_hardsigmoid"]) {
    res = test_hardsigmoid();
  } else if ([testName isEqualToString:@"test_hardswish"]) {
    res = test_hardswish();
  } else if ([testName isEqualToString:@"test_upsampling_nearest2d_vec"]) {
    res = test_upsampling_nearest2d_vec();
  } else if ([testName isEqualToString:@"test_adaptive_avg_pool2d"]) {
    res = test_adaptive_avg_pool2d();
  } else if ([testName isEqualToString:@"test_hardtanh_"]) {
    res = test_hardtanh_();
  } else if ([testName isEqualToString:@"test_reshape"]) {
    res = test_reshape();
  } else if ([testName isEqualToString:@"test_mean_dim"]) {
    res = test_mean_dim();
  } else if ([testName isEqualToString:@"test_mean_dim2"]) {
    res = test_mean_dim2();
  } else if ([testName isEqualToString:@"test_mean_dim3"]) {
    res = test_mean_dim3();
  }
  return res ? @(1) : @(0);
}

@end
