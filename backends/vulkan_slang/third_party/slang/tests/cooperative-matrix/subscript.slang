//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type  -emit-spirv-directly -skip-spirv-validation

// CHECK: type: int32_t
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK: 7
// CHECK-NEXT: 11

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWStructuredBuffer<int32_t> outputBuffer;

typealias CoopMatType = CoopMat<int32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain()
{
    CoopMatType mat;
    mat[0] = 2;
    mat[1] = mat[0]+2;
    mat[2] = mat[1]+3;
    mat[3] = mat[2]+4;
    mat.store(outputBuffer, 0, 16, CoopMatMatrixLayout::RowMajor);
}
