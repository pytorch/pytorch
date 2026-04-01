//DIAGNOSTIC_TEST(compute):SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target spirv

RWStructuredBuffer<float> outputBuffer;

typealias CoopMatAType = CoopMat<float16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixA>;
typealias CoopMatBType = CoopMat<float16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixA>;
typealias CoopMatCType = CoopMat<float32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

// CHECK: error 41400: static assertion failed, matrix uses for `coopMatMulAdd` matrix parameters must be `MatrixA`, `MatrixB` and `MatrixAccumulator`

[numthreads(32, 1, 1)]
void computeMain()
{
    let matA = CoopMatAType(3.0);
    let matB = CoopMatBType(5.0);
    let matC = CoopMatCType(1.0);

    let result = coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::None);
    result.store(outputBuffer, 0, 16, CoopMatMatrixLayout::RowMajor);
}
