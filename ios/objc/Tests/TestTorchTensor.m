#import <XCTest/XCTest.h>
#import <PytorchObjC/PytorchObjC.h>

@interface TestTorchTensor : XCTestCase

@end

@implementation TestTorchTensor{
    TorchModule* _module;
}

- (void)setUp {
    NSString* filePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"test" ofType:@"pt"];
    _module = [TorchModule loadTorchscriptModel:filePath];
}

- (void)testTensorSize {
    //creat a 3x2 tensor
    int32_t t[3][2] = {{1,2},{3,4},{4,6}};
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(3),@(2)] Data:t]; //2x2 tensor
    XCTAssertEqual(tensor.type, TorchTensorTypeInt);
    NSArray* size = @[@(3),@(2)]; //3x2
    XCTAssertEqualObjects(tensor.size, size);
    size = @[@(2)]; //1x2
    XCTAssertEqualObjects(tensor[0].size, size);
    XCTAssertEqualObjects(tensor[1].size, size);
    XCTAssertEqualObjects(tensor[2].size, size);
    size = @[];
    XCTAssertEqualObjects(tensor[0][0].size, size);
    XCTAssertEqualObjects(tensor[0][1].size, size);
    XCTAssertEqualObjects(tensor[1][0].size, size);
    XCTAssertEqualObjects(tensor[1][1].size, size);
    XCTAssertEqualObjects(tensor[2][0].size, size);
    XCTAssertEqualObjects(tensor[2][1].size, size);
}

- (void)testTensorDim {
    //creat a 2x2 tensor
    int32_t t[2][2] = {{1,1},{1,1}};
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t]; //2x2 tensor
    XCTAssertEqual(tensor.type, TorchTensorTypeInt);
    XCTAssertEqual(tensor.dim, 2);
    XCTAssertEqual(tensor[0].dim, 1);
    XCTAssertEqual(tensor[1].dim, 1);
    XCTAssertEqual(tensor[0][0].dim, 0);
    XCTAssertEqual(tensor[0][1].dim, 0);
    XCTAssertEqual(tensor[1][0].dim, 0);
    XCTAssertEqual(tensor[1][1].dim, 0);
}

- (void)testTensorValue {
    //creat a 2x2 tensor
    int32_t t[2][2] = {{1,2},{3,4}};
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t]; //2x2 tensor
    XCTAssertEqual(tensor[0][0].item.integerValue, 1);
    XCTAssertEqual(tensor[0][1].item.integerValue, 2);
    XCTAssertEqual(tensor[1][0].item.integerValue, 3);
    XCTAssertEqual(tensor[1][1].item.integerValue, 4);
}

- (void)testTensorToType {
    //creat a 2x2 tensor
    int32_t t[2][2] = {{1,2},{3,4}};
    TorchTensor* intTensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t]; //2x2 tensor
    TorchTensor* floatTensor = [intTensor to:TorchTensorTypeFloat];
    XCTAssertEqual(floatTensor[0][0].item.floatValue, 1.0);
    XCTAssertEqual(floatTensor[0][1].item.floatValue, 2.0);
    XCTAssertEqual(floatTensor[1][0].item.floatValue, 3.0);
    XCTAssertEqual(floatTensor[1][1].item.floatValue, 4.0);
}

- (void)testTensorItem {
    //creat a 2x2 tensor
    int32_t t[2][2] = {{1,2},{3,4}};
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(2)] Data:t]; //2x2 tensor
    XCTAssertNil(tensor.item); //2x2 tensor
    XCTAssertNil(tensor[0].item); //1x2 tensor
    XCTAssertEqual(tensor[0][0].item.integerValue, 1);
    XCTAssertEqual(tensor[0][1].item.integerValue, 2);
    XCTAssertEqual(tensor[1][0].item.integerValue, 3);
    XCTAssertEqual(tensor[1][1].item.integerValue, 4);
}

- (void)testTensorPermute {
    //creat a 1x2x3 tensor
    int32_t t[1][2][3] ={{{1,2,3},{4,5,6}}};
    NSArray* size = @[@(1),@(2),@(3)];
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:size Data:t]; //1x2x3 tensor
    TorchTensor* tensor1 = [tensor permute:@[@(0),@(1),@(2)]];
    XCTAssertEqualObjects(tensor1.size, size);
    TorchTensor* tensor2 = [tensor permute:@[@(0),@(2),@(1)]]; //1x3x2
    size = @[@(1),@(3),@(2)];
    XCTAssertEqualObjects(tensor2.size, size);
    TorchTensor* tensor3 = [tensor permute:@[@(1),@(0),@(2)]]; //2x1x3
    size = @[@(2),@(1),@(3)];
    XCTAssertEqualObjects(tensor3.size, size);
    TorchTensor* tensor4 = [tensor permute:@[@(1),@(2),@(0)]]; //2x3x1
    size = @[@(2),@(3),@(1)];
    XCTAssertEqualObjects(tensor4.size, size);
    TorchTensor* tensor5 = [tensor permute:@[@(2),@(1),@(0)]]; //3x2x1
    size = @[@(3),@(2),@(1)];
    XCTAssertEqualObjects(tensor5.size, size);
    TorchTensor* tensor6 = [tensor permute:@[@(2),@(0),@(1)]]; //3x1x2
    size = @[@(3),@(1),@(2)];
    XCTAssertEqualObjects(tensor6.size, size);
}

- (void)testTensorView {
    //creat a 2x2 tensor
    int32_t t[2][3] = {{1,1,1},{1,1,1}};
    TorchTensor* tensor = [TorchTensor newWithType:TorchTensorTypeInt Size:@[@(2),@(3)] Data:t]; //2x2 tensor
    TorchTensor* tensor1 = [tensor view:@[@(1),@(-1)]]; //1x6
    NSArray* size = @[@(1),@(6)];
    XCTAssertEqualObjects(tensor1.size, size);
    TorchTensor* tensor2 = [tensor view:@[@(6),@(-1)]]; //6x1
    size = @[@(6),@(1)];
    XCTAssertEqualObjects(tensor2.size, size);
    TorchTensor* tensor3 = [tensor view:@[@(2),@(-1)]]; //2x3
    size = @[@(2),@(3)];
    XCTAssertEqualObjects(tensor3.size, size);
    TorchTensor* tensor4 = [tensor view:@[@(3),@(-1)]]; //3x2
    size = @[@(3),@(2)];
    XCTAssertEqualObjects(tensor4.size, size);
}

@end
