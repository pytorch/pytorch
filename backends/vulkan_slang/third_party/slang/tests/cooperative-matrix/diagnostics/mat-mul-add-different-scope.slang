//DIAGNOSTIC_TEST(compute):SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target spirv

RWStructuredBuffer<float> outputBuffer;

typealias CoopMatAType = CoopMat<float16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixA>;
typealias CoopMatBType = CoopMat<float16_t, CoopMatScope::Workgroup, 16, 16, CoopMatMatrixUse::MatrixB>;
typealias CoopMatCType = CoopMat<float32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

// CHECK: error 39999: could not specialize generic for arguments of type

[numthreads(32, 1, 1)]
void computeMain()
{
    let matA = CoopMatAType(3.0);
    let matB = CoopMatBType(5.0);
    let matC = CoopMatCType(1.0);

    const let result = coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::None);
    result.store(outputBuffer, 0, 16, CoopMatMatrixLayout::RowMajor);
}
