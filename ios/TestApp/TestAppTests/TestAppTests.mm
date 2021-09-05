#import <XCTest/XCTest.h>

#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
}

- (void)testLiteInterpreter {
    NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model"
                                                                           ofType:@"ptl"];
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:modelPath],
                  @"model doesn't exist!");
    XCTAssertTrue([self runModel:modelPath], @"");
}

- (void)testCoreML {
    NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"coreml"
                                                                           ofType:@"ptl"];
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:modelPath],
                  @"model doesn't exist!");
    XCTAssertTrue([self runModel:modelPath], @"");
}

- (bool)runModel:(NSString* )path {
    torch::jit::mobile::Module module = torch::jit::_load_for_mobile(path.UTF8String);
    c10::InferenceMode mode;
    std::vector<c10::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}, at::ScalarType::Float));
    auto outputTensor = module.forward(inputs).toTensor();
    return outputTensor.numel() == 1000;
}

@end
