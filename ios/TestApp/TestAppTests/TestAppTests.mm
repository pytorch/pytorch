#import <XCTest/XCTest.h>
#import <torch/script.h>

@interface TestAppTests : XCTestCase

@end

@implementation TestAppTests {
  torch::jit::script::Module _module;
}

- (void)setUp {
  [super setUp];
  NSString* modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"model"
                                                                         ofType:@"pt"];
  XCTAssertTrue([[NSFileManager defaultManager] fileExistsAtPath:modelPath],
                @"model.pt doesn't exist!");
  auto qengines = at::globalContext().supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
    at::globalContext().setQEngine(at::QEngine::QNNPACK);
  }
  torch::autograd::AutoGradMode guard(false);
  _module = torch::jit::load(std::string(modelPath.UTF8String));
  _module.eval();
}

- (void)tearDown {
  [super tearDown];
}

- (void)testForward {
  std::vector<c10::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}, at::ScalarType::Float));
  auto outputTensor = _module.forward(inputs).toTensor().view(-1);
  XCTAssertEqual(outputTensor.numel(), 1000);
}

@end
