//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -skip-spirv-validation

// CHECK: type: float
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 9.000000
// CHECK-NEXT: 12.000000

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4, count=256):name=inputBuffer
StructuredBuffer<float> inputBuffer;

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typealias CoopMatType = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

CoopMatType doubleCoopmat(CoopMatType mat)
{
    return mat * 3.0;
}

[numthreads(32, 1, 1)]
void computeMain()
{
    let stride = 16;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let mat = CoopMatType.load(inputBuffer, 0, stride, matrixLayout);

    let result = doubleCoopmat(mat);
    result.store(outputBuffer, 0, 4, matrixLayout);
}
