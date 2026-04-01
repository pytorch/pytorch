//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -emit-spirv-directly -skip-spirv-validation

// CHECK: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 9
// CHECK-NEXT: A
// CHECK-NEXT: B
// CHECK-NEXT: C
// CHECK-NEXT: D
// CHECK-NEXT: E
// CHECK-NEXT: F
// CHECK-NEXT: 10

//TEST_INPUT:ubuffer(data=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16], stride=4, count=256):name=input
RWByteAddressBuffer input;

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<uint32_t> outputBuffer;

typealias CoopMatType = CoopMat<uint32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

groupshared float[256] tempShared;

[numthreads(32, 1, 1)]
void computeMain()
{
    let stride = 16;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let mat = coopMatLoad<uint32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(input, 0, stride, matrixLayout);
    mat.storeAny(tempShared, 0, stride, matrixLayout);

    let result = CoopMatType.loadAny(tempShared, 0, stride, matrixLayout);
    result.store(outputBuffer, 0, stride, matrixLayout);
}
