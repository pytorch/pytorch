//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: float
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 8.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4, count=256),name=input
ByteAddressBuffer input;


[numthreads(32, 1, 1)]
void computeMain()
{
    let stride = 16;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let intMat = CoopMat<int, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>.load(input, 0, stride, matrixLayout);
    let floatMat = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(intMat);
    let uintMat = CoopMat<uint, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(intMat);
    let halfMat = CoopMat<half, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(uintMat);
    let floatMat2 = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(halfMat);

    let result = floatMat + floatMat2;
    result.store(outputBuffer, 0, stride, matrixLayout);
}
