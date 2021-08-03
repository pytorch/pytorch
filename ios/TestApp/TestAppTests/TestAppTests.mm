#import <XCTest/XCTest.h>

#include <torch/script.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>
#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
  torch::jit::mobile::Module _module;
}

+ (void)setUp {
  [super setUp];
}

- (void)setUp {
  [super setUp];
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model"
                                                                         ofType:@"ptl"];
  XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:modelPath],
                @"model.ptl doesn't exist!");
  _module = torch::jit::_load_for_mobile(modelPath.UTF8String);
}

- (void)testForward {
//  _module.eval();
  c10::InferenceMode mode;
  std::vector<c10::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}, at::ScalarType::Float));
  auto outputTensor = _module.forward(inputs).toTensor();
  float* outputBuffer = outputTensor.data_ptr<float>();
  XCTAssertTrue(outputBuffer != nullptr, @"");
}

@end
