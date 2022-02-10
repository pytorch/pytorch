#import <XCTest/XCTest.h>

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
}

- (void)testMobileNetV2 {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"mobilenet_v2"
                                                                         ofType:@"ptl"];
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  c10::InferenceMode mode;
  auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
  auto outputTensor = module.forward({input}).toTensor();
  XCTAssertTrue(outputTensor.numel() == 1000);
}

- (void)testMobileNetV3 {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"mobilenet_v3_small"
                                                                         ofType:@"ptl"];
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  c10::InferenceMode mode;
  auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
  auto outputTensor = module.forward({input}).toTensor();
  XCTAssertTrue(outputTensor.numel() == 1000);
}

- (void)testKeypointrcnn {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"keypointrcnn_resnet50_fpn"
                                                                         ofType:@"ptl"];
  auto module = torch::jit::_load_for_mobile(modelPath.UTF8String);
  c10::InferenceMode mode;
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(torch::rand(3, 300, 400));
  inputs.emplace_back(torch::rand(3, 400, 500));
  auto outputTensor = module.forward({inputs}).toTensor();
  XCTAssertTrue(true);
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


@end
