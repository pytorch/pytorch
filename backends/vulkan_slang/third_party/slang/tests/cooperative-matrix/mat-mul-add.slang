//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: float
// CHECK-COUNT-256: 241.0

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typealias CoopMatAType = CoopMat<float16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixA>;
typealias CoopMatBType = CoopMat<float16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixB>;
typealias CoopMatCType = CoopMat<float32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain()
{
    // ( 3.0 * 5.0 ) * 16 + 1.0 = 241.0
    let matA = CoopMatAType(3.0);
    let matB = CoopMatBType(5.0);
    let matC = CoopMatCType(1.0);

    let result = coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::None);
    result.store(outputBuffer, 0, 16, CoopMatMatrixLayout::RowMajor);
}
