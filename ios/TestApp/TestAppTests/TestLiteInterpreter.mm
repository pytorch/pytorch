#import <XCTest/XCTest.h>

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
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

@end
