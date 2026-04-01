//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: int32_t
// CHECK-COUNT-256: 4

//TEST_INPUT:ubuffer(stride=4, count = 256):out,name=outputBuffer
RWStructuredBuffer<int32_t> outputBuffer;

[numthreads(32, 1, 1)]
void computeMain()
{
    let mat = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(4.0);
    var result = CoopMat<int32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>(0);
    result.copyFrom(mat);
    result.store(outputBuffer, 0, 16, CoopMatMatrixLayout::RowMajor);
}

