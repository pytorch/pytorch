//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: int32_t
// CHECK-NEXT: -1
// CHECK-NEXT: -2
// CHECK-NEXT: -3
// CHECK-NEXT: -4

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4, count=256),name=inputBuffer
ByteAddressBuffer inputBuffer;

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<int32_t> outputBuffer;

typealias CoopMatType = CoopMat<int32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain()
{
    let stride = 4;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let mat = CoopMatType.load(inputBuffer, 0, stride, matrixLayout);

    let result = -mat;
    result.store(outputBuffer, 0, 4, matrixLayout);
}
