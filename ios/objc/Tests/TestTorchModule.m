#import <XCTest/XCTest.h>
#import <PytorchObjC/PytorchObjC.h>

@interface TestTorchModule : XCTestCase

@end

@implementation TestTorchModule{
    TorchModule* _module;
}

- (void)setUp {
    NSString* filePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"test" ofType:@"pt"];
    _module = [TorchModule loadTorchscriptModel:filePath];
}

- (void)testForward {
    int32_t t1[2][2] = {{1,1},{1,1}};
    int32_t t2[2][2] = {{2,2},{2,2}};
    TorchTensor* tensor1 = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t1]; //2x2 tensor
    TorchTensor* tensor2 = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t2]; //2x2 tensor
    TorchIValue* input1  = [TorchIValue newWithTensor:tensor1];
    TorchIValue* input2  = [TorchIValue newWithTensor:tensor2];
    //(Tensor, Tensor) -> Tensor
    TorchIValue* output = [_module forward:@[input1, input2]];
    TorchIValueType type = output.type;
    XCTAssertEqual(type, TorchIValueTypeTensor);
    TorchTensor* outputTensor = output.toTensor;
    XCTAssertEqual(outputTensor[0][0].item.integerValue, 3);
    XCTAssertEqual(outputTensor[0][1].item.integerValue, 3);
    XCTAssertEqual(outputTensor[1][0].item.integerValue, 3);
    XCTAssertEqual(outputTensor[1][1].item.integerValue, 3);
}

- (void)testRunMethod {
    TorchIValue* input  = [TorchIValue newWithBool:@(YES)];
    //(bool) -> bool
    TorchIValue* output = [_module run_method:@"eqBool" withInputs:@[input]];
    TorchIValueType type = output.type;
    XCTAssertEqual(type, TorchIValueTypeBool);
    XCTAssertEqual(output.toBool.boolValue, YES);
}

@end
