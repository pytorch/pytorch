#import <XCTest/XCTest.h>
#import <PytorchObjC/PytorchObjC.h>

@interface TestTorchIValue : XCTestCase

@end

@implementation TestTorchIValue{
    TorchModule* _module;
}

- (void)setUp {
    NSString* filePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"test" ofType:@"pt"];
    _module = [TorchModule loadTorchscriptModel:filePath];
}


- (void)testBool {
    TorchIValue* input  = [TorchIValue newWithBool:@(YES)];
    //(bool) -> bool
    TorchIValue* output = [_module run_method:@"eqBool" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeBool);
    XCTAssertEqual(output.toBool.boolValue, YES);
}


- (void)testInt {
    TorchIValue* input  = [TorchIValue newWithInt:@(-1)];
    //(int) -> int
    TorchIValue* output = [_module run_method:@"eqInt" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeInt);
    XCTAssertEqual(output.toInt.integerValue, -1);
}


- (void)testDouble {
    TorchIValue* input  = [TorchIValue newWithDouble:@(1.0)];
    //(double) -> double
    TorchIValue* output = [_module run_method:@"eqDouble" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeDouble);
    XCTAssertEqual(output.toDouble.doubleValue, 1.0);
}

- (void)testTensor {
    int32_t t[2][2] = {{1,1},{1,1}};
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t]; //2x2 tensor
    TorchIValue* input  = [TorchIValue newWithTensor:tensor];
    //(tensor) -> tensor
    TorchIValue* output = [_module run_method:@"eqTensor" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeTensor);
    TorchTensor* outputTensor = output.toTensor;
    XCTAssertEqual(outputTensor[0][0].item.integerValue, 1);
    XCTAssertEqual(outputTensor[0][1].item.integerValue, 1);
    XCTAssertEqual(outputTensor[1][0].item.integerValue, 1);
    XCTAssertEqual(outputTensor[1][1].item.integerValue, 1);
}

- (void)testBoolList {
    TorchIValue* input  = [TorchIValue newWithBoolList:@[@(YES), @(NO)]];
    //(list[bool]) -> (list[bool])
    TorchIValue* output = [_module run_method:@"eqBoolList" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeBoolList);
    XCTAssertEqual(output.toBoolList[0].boolValue, YES);
    XCTAssertEqual(output.toBoolList[1].boolValue, NO);
}

- (void)testIntList {
    TorchIValue* input  = [TorchIValue newWithIntList:@[@(1), @(1), @(1)]];
    //(list[Int]) -> (list[Int])
    TorchIValue* output = [_module run_method:@"eqIntList" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeIntList);
    XCTAssertEqual(output.toIntList[0].integerValue, 1);
    XCTAssertEqual(output.toIntList[1].integerValue, 1);
    XCTAssertEqual(output.toIntList[2].integerValue, 1);
}

- (void)testDoubleList {
    TorchIValue* input  = [TorchIValue newWithDoubleList:@[@(0.1), @(0.1), @(0.1)]];
    //(list[Double]) -> (list[Double])
    TorchIValue* output = [_module run_method:@"eqDoubleList" withInputs:@[input]];
    XCTAssertEqual(output.type, TorchIValueTypeDoubleList);
    XCTAssertEqual(output.toDoubleList[0].doubleValue, 0.1);
    XCTAssertEqual(output.toDoubleList[1].doubleValue, 0.1);
    XCTAssertEqual(output.toDoubleList[2].doubleValue, 0.1);
}

- (void)testTensorList {
    int32_t t1[2][2] = {{1,1},{1,1}};
    int32_t t2[2][2] = {{2,2},{2,2}};
    TorchTensor* tensor1 = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t1]; //2x2 tensor
    TorchTensor* tensor2 = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t2]; //2x2 tensor
    TorchIValue* input   = [TorchIValue newWithTensorList:@[tensor1,tensor2]];
    //(list[Tensor]) -> (list[Tensor])
    TorchIValue* output = [_module run_method:@"eqTensorList" withInputs:@[input]];
    XCTAssertEqual( output.type, TorchIValueTypeTensorList);
    NSArray<TorchTensor*>* outputTensorList = output.toTensorList;
    XCTAssertEqual(outputTensorList[0][0][0].item.integerValue, 1);
    XCTAssertEqual(outputTensorList[0][0][1].item.integerValue, 1);
    XCTAssertEqual(outputTensorList[0][1][0].item.integerValue, 1);
    XCTAssertEqual(outputTensorList[0][1][1].item.integerValue, 1);
    XCTAssertEqual(outputTensorList[1][0][0].item.integerValue, 2);
    XCTAssertEqual(outputTensorList[1][0][1].item.integerValue, 2);
    XCTAssertEqual(outputTensorList[1][1][0].item.integerValue, 2);
    XCTAssertEqual(outputTensorList[1][1][1].item.integerValue, 2);
}

@end
