#import <XCTest/XCTest.h>

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
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

- (void)testModel:(NSString*)modelName {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:modelName
                                                                         ofType:@"ptl"];
  XCTAssertNotNil(modelPath, @"Model not found. See https://github.com/pytorch/pytorch/tree/master/test/mobile/model_test#diagnose-failed-test.");
  [self runModel:modelPath];

  // model generated on the fly
  NSString* onTheFlyModelName = [NSString stringWithFormat:@"%@", modelName];
  NSString* onTheFlyModelPath = [[NSBundle bundleForClass:[self class]] pathForResource:onTheFlyModelName
                                                                         ofType:@"ptl"];
  XCTAssertNotNil(onTheFlyModelPath, @"On-the-fly model not found. Follow https://github.com/pytorch/pytorch/tree/master/test/mobile/model_test#diagnose-failed-test to generate them and run the setup.rb script again.");
  [self runModel:onTheFlyModelPath];
}

- (void)runModel:(NSString*)modelPath {
  c10::InferenceMode mode;
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  auto has_bundled_input = module.find_method("get_all_bundled_inputs");
  if (has_bundled_input) {
    c10::IValue bundled_inputs = module.run_method("get_all_bundled_inputs");
    c10::List<at::IValue> all_inputs = bundled_inputs.toList();
    std::vector<std::vector<at::IValue>> inputs;
    for (at::IValue input : all_inputs) {
      inputs.push_back(input.toTupleRef().elements());
    }
    // run with the first bundled input
    XCTAssertNoThrow(module.forward(inputs[0]));
  } else {
    XCTAssertNoThrow(module.forward({}));
  }
}

// TODO remove this once updated test script
- (void)testLiteInterpreter {
  XCTAssertTrue(true);
}

- (void)testMobileNetV2 {
  [self testModel:@"mobilenet_v2"];
}

- (void)testPointwiseOps {
  [self testModel:@"pointwise_ops"];
}

- (void)testReductionOps {
  [self testModel:@"reduction_ops"];
}

- (void)testComparisonOps {
  [self testModel:@"comparison_ops"];
}

- (void)testOtherMathOps {
  [self testModel:@"other_math_ops"];
}

- (void)testSpectralOps {
  [self testModel:@"spectral_ops"];
}

- (void)testBlasLapackOps {
  [self testModel:@"blas_lapack_ops"];
}

- (void)testSamplingOps {
  [self testModel:@"sampling_ops"];
}

- (void)testTensorOps {
  [self testModel:@"tensor_general_ops"];
}

- (void)testTensorCreationOps {
  [self testModel:@"tensor_creation_ops"];
}

- (void)testTensorIndexingOps {
  [self testModel:@"tensor_indexing_ops"];
}

- (void)testTensorTypingOps {
  [self testModel:@"tensor_typing_ops"];
}

- (void)testTensorViewOps {
  [self testModel:@"tensor_view_ops"];
}

- (void)testConvolutionOps {
  [self testModel:@"convolution_ops"];
}

- (void)testPoolingOps {
  [self testModel:@"pooling_ops"];
}

- (void)testPaddingOps {
  [self testModel:@"padding_ops"];
}

- (void)testActivationOps {
  [self testModel:@"activation_ops"];
}

- (void)testNormalizationOps {
  [self testModel:@"normalization_ops"];
}

- (void)testRecurrentOps {
  [self testModel:@"recurrent_ops"];
}

- (void)testTransformerOps {
  [self testModel:@"transformer_ops"];
}

- (void)testLinearOps {
  [self testModel:@"linear_ops"];
}

- (void)testDropoutOps {
  [self testModel:@"dropout_ops"];
}

- (void)testSparseOps {
  [self testModel:@"sparse_ops"];
}

- (void)testDistanceFunctionOps {
  [self testModel:@"distance_function_ops"];
}

- (void)testLossFunctionOps {
  [self testModel:@"loss_function_ops"];
}

- (void)testVisionFunctionOps {
  [self testModel:@"vision_function_ops"];
}

- (void)testShuffleOps {
  [self testModel:@"shuffle_ops"];
}

- (void)testNNUtilsOps {
  [self testModel:@"nn_utils_ops"];
}

- (void)testQuantOps {
  [self testModel:@"general_quant_ops"];
}

- (void)testDynamicQuantOps {
  [self testModel:@"dynamic_quant_ops"];
}

- (void)testStaticQuantOps {
  [self testModel:@"static_quant_ops"];
}

- (void)testFusedQuantOps {
  [self testModel:@"fused_quant_ops"];
}

- (void)testTorchScriptBuiltinQuantOps {
  [self testModel:@"torchscript_builtin_ops"];
}

- (void)testTorchScriptCollectionQuantOps {
  [self testModel:@"torchscript_collection_ops"];
}

@end
