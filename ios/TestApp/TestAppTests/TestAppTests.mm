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
  XCTAssertTrue([self runModel:modelPath], @"");
}

- (void)testFullJIT {
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model"
                                                                         ofType:@"pt"];
  XCTAssertTrue([self runModel:modelPath], @"");
}

- (bool)runModel:(NSString*)path {
  XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:path], @"model doesn't exist!");
  torch::jit::mobile::Module module = torch::jit::_load_for_mobile(path.UTF8String);
  c10::InferenceMode mode;
  auto input = torch::ones({1, 3, 224, 224}, at::kFloat);
  auto outputTensor = module.forward({input}).toTensor();
  return outputTensor.numel() == 1000;
}

@end
