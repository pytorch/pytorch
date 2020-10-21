#import <XCTest/XCTest.h>

#include <ATen/native/metal/mpscnn/tests/MPSCNNTests.h>
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

- (void)testMetalOps {
#ifdef USE_PYTORCH_METAL
  if (@available(iOS 10.0, *)) {
    if ([[MPSCNNContext sharedInstance] available]) {
      using namespace at::native::metal;
      XCTAssert(test_aten(), @"Test aten failed!");
      XCTAssert(test_NC4(), @"Test NC4 failed!");
      XCTAssert(test_MPSImage(), @"Test MPSImage failed!");
      XCTAssert(test_MPSImageCopy(), @"Test MPSImageCopy failed!");
      XCTAssert(test_MPSTemporaryImageCopy(), @"Test MPSTemporaryImageCopy failed!");
      XCTAssert(test_conv2d(), @"Test conv2d failed!");
      XCTAssert(test_depthwiseConv(), @"Test depthwiseConv failed!");
      XCTAssert(test_max_pool2d(), @"Test max_pool2d failed!");
      XCTAssert(test_relu(), @"Test relu failed!");
      XCTAssert(test_addmm(), @"Test addmm failed!");
      XCTAssert(test_add(), @"Test add failed!");
      XCTAssert(test_sub(), @"Test sub failed!");
      XCTAssert(test_mul(), @"Test mul failed!");
      XCTAssert(test_t(), @"Test transpose2d failed!");
      XCTAssert(test_view(), @"Test view failed!");
      XCTAssert(test_softmax(), @"Test softmax failed!");
      XCTAssert(test_sigmoid(), @"Test sigmoid failed!");
      XCTAssert(test_upsampling_nearest2d_vec(), @"Test upsampling_nearest2d failed!");
      XCTAssert(test_adaptive_avg_pool2d(), @"Test adaptive_avg_pool2d failed!");
      XCTAssert(test_hardtanh_(), @"Test hardtanh failed!");
      XCTAssert(test_reshape(), @"Test reshape failed!");
    } else {
      FBLogInfo(@"Metal is not available!");
    }
  }
#endif
}

@end
