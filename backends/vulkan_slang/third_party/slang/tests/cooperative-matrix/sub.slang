//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: int32_t
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<int32_t> outputBuffer;

//TEST_INPUT:ubuffer(data=[2 3 4 5], stride=4, count=256),name=input1
ByteAddressBuffer input1;

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4, count=256),name=input2
ByteAddressBuffer input2;

typealias CoopMatType = CoopMat<int32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain()
{
    let stride = 16;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let mat1 = CoopMatType.load(input1, 0, stride, matrixLayout);
    let mat2 = CoopMatType.load(input2, 0, stride, matrixLayout);

    let result = mat1 - mat2;
    result.store(outputBuffer, 0, 4, matrixLayout);
}
