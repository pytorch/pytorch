//TEST(compute):SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target spirv -skip-spirv-validation

// This test checks that the correct SPIRV Cooperative Matrix Operands are emitted for OpCooperativeMatrixMulAddKHR operaions
RWStructuredBuffer<int> outputBuffer1;
RWStructuredBuffer<int> outputBuffer2;
RWStructuredBuffer<int> outputBuffer3;
RWStructuredBuffer<int> outputBuffer4;
RWStructuredBuffer<int> outputBuffer5;

typealias CoopMatAType = CoopMat<int16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixA>;
typealias CoopMatBType = CoopMat<int16_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixB>;
typealias CoopMatCType = CoopMat<int32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain()
{
    let matA = CoopMatAType(3);
    let matB = CoopMatBType(5);
    let matC = CoopMatCType(1);

    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    // CHECK: OpCooperativeMatrixMulAddKHR {{.*}} NoneKHR
    coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::None).store(outputBuffer1, 0, 16, matrixLayout);

    // CHECK: OpCooperativeMatrixMulAddKHR {{.*}} MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR
    coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::MatrixASigned | CoopMatMatrixOperands::MatrixBSigned).store(outputBuffer2, 0, 16, matrixLayout);


    // CHECK: OpCooperativeMatrixMulAddKHR {{.*}} MatrixCSignedComponentsKHR
    coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::MatrixCSigned).store(outputBuffer2, 0, 16, matrixLayout);


    // CHECK: OpCooperativeMatrixMulAddKHR {{.*}} MatrixResultSignedComponentsKHR
    coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::MatrixResultSigned).store(outputBuffer3, 0, 16, matrixLayout);

    // CHECK: OpCooperativeMatrixMulAddKHR {{.*}} SaturatingAccumulationKHR
    coopMatMulAdd(matA, matB, matC, CoopMatMatrixOperands::SaturatingAccumulation).store(outputBuffer4, 0, 16, matrixLayout);

    let allOperands = CoopMatMatrixOperands::MatrixASigned | CoopMatMatrixOperands::MatrixBSigned | CoopMatMatrixOperands::MatrixCSigned | CoopMatMatrixOperands::MatrixResultSigned | CoopMatMatrixOperands::SaturatingAccumulation;
    // CHECK: OpCooperativeMatrixMulAddKHR {{.*}} MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR|MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR|SaturatingAccumulationKHR
    coopMatMulAdd(matA, matB, matC, allOperands).store(outputBuffer5, 0, 16, matrixLayout);
}


