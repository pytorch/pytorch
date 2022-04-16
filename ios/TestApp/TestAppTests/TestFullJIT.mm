#import <XCTest/XCTest.h>

#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
}

- (void)testFullJIT {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model"
                                                                         ofType:@"pt"];
  auto module = torch::jit::load(modelPath.UTF8String);
  c10::InferenceMode mode;
  auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
  auto outputTensor = module.forward({input}).toTensor();
  XCTAssertTrue(outputTensor.numel() == 1000);
}

@end
