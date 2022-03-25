#import <XCTest/XCTest.h>

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
}

- (void)runModel:(NSString*)filename {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:filename
                                                                         ofType:@"ptl"];
  XCTAssertNotNil(modelPath);
  c10::InferenceMode mode;
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  XCTAssertNoThrow(module.forward({}));
}

- (void)testLiteInterpreter {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model_lite"
                                                                         ofType:@"ptl"];
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  c10::InferenceMode mode;
  auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
  auto outputTensor = module.forward({input}).toTensor();
  XCTAssertTrue(outputTensor.numel() == 1000);
}

- (void)testCoreML {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model_coreml"
                                                                         ofType:@"ptl"];
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  c10::InferenceMode mode;
  auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
  auto outputTensor = module.forward({input}).toTensor();
  XCTAssertTrue(outputTensor.numel() == 1000);
}

- (void)testPointwiseOps {
  [self runModel:@"pointwise_ops"];
}

- (void)testReductionOps {
  [self runModel:@"reduction_ops"];
}

- (void)testComparisonOps {
  [self runModel:@"comparison_ops"];
}

- (void)testOtherMathOps {
  [self runModel:@"other_math_ops"];
}

- (void)testSpectralOps {
  [self runModel:@"spectral_ops"];
}

- (void)testBlasLapackOps {
  [self runModel:@"blas_lapack_ops"];
}

- (void)testSamplingOps {
  [self runModel:@"sampling_ops"];
}

- (void)testTensorOps {
  [self runModel:@"tensor_general_ops"];
}

- (void)testTensorCreationOps {
  [self runModel:@"tensor_creation_ops"];
}

- (void)testTensorIndexingOps {
  [self runModel:@"tensor_indexing_ops"];
}

- (void)testTensorTypingOps {
  [self runModel:@"tensor_typing_ops"];
}

- (void)testTensorViewOps {
  [self runModel:@"tensor_view_ops"];
}

- (void)testConvolutionOps {
  [self runModel:@"convolution_ops"];
}

- (void)testPoolingOps {
  [self runModel:@"pooling_ops"];
}

- (void)testPaddingOps {
  [self runModel:@"padding_ops"];
}

- (void)testActivationOps {
  [self runModel:@"activation_ops"];
}

- (void)testNormalizationOps {
  [self runModel:@"normalization_ops"];
}

- (void)testRecurrentOps {
  [self runModel:@"recurrent_ops"];
}

- (void)testTransformerOps {
  [self runModel:@"transformer_ops"];
}

- (void)testLinearOps {
  [self runModel:@"linear_ops"];
}

- (void)testDropoutOps {
  [self runModel:@"dropout_ops"];
}

- (void)testSparseOps {
  [self runModel:@"sparse_ops"];
}

- (void)testDistanceFunctionOps {
  [self runModel:@"distance_function_ops"];
}

- (void)testLossFunctionOps {
  [self runModel:@"loss_function_ops"];
}

- (void)testVisionFunctionOps {
  [self runModel:@"vision_function_ops"];
}

- (void)testShuffleOps {
  [self runModel:@"shuffle_ops"];
}

- (void)testNNUtilsOps {
  [self runModel:@"nn_utils_ops"];
}

- (void)testQuantOps {
  [self runModel:@"general_quant_ops"];
}

- (void)testDynamicQuantOps {
  [self runModel:@"dynamic_quant_ops"];
}

- (void)testStaticQuantOps {
  [self runModel:@"static_quant_ops"];
}

- (void)testFusedQuantOps {
  [self runModel:@"fused_quant_ops"];
}

- (void)testTorchScriptBuiltinQuantOps {
  [self runModel:@"torchscript_builtin_ops"];
}

- (void)testTorchScriptCollectionQuantOps {
  [self runModel:@"torchscript_collection_ops"];
}
@end
