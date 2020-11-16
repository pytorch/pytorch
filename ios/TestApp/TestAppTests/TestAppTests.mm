#import <XCTest/XCTest.h>

#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
  torch::jit::Module _module;
}

+ (void)setUp {
  [super setUp];
}

- (void)setUp {
  [super setUp];
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model"
                                                                         ofType:@"pt"];
  XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:modelPath],
                @"model.pt doesn't exist!");
  _module = torch::jit::load(modelPath.UTF8String);
}

- (void)testForward {
  _module.eval();
  std::vector<c10::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}, at::ScalarType::Float));
  torch::autograd::AutoGradMode guard(false);
  at::AutoNonVariableTypeMode nonVarTypeModeGuard(true);
  auto outputTensor = _module.forward(inputs).toTensor();
  float* outputBuffer = outputTensor.data_ptr<float>();
  XCTAssertTrue(outputBuffer != nullptr, @"");
}

@end
