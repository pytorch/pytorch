//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -emit-spirv-directly -skip-spirv-validation

// CHECK: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2

// CHECK: 0
// CHECK-NEXT: 0
// CHECK: 1
// CHECK-NEXT: 2

// CHECK: 0
// CHECK-NEXT: 0
// CHECK: 1
// CHECK-NEXT: 2

//TEST_INPUT:ubuffer(stride=4, count=256):out,name=outputBuffer
RWByteAddressBuffer outputBuffer;

//TEST_INPUT:ubuffer(data=[4 3 5 7], stride=4, count=256),name=input1
ByteAddressBuffer input1;

//TEST_INPUT:ubuffer(data=[2 3 4 5], stride=4, count=256),name=input2
ByteAddressBuffer input2;

typealias CoopMatIntType = CoopMat<int32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;
typealias CoopMatUintType = CoopMat<uint32_t, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;
typealias CoopMatFloatType = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain()
{
    let stride = 16;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let mat1 = CoopMatIntType.load(input1, 0, stride, matrixLayout);
    let mat2 = CoopMatIntType.load(input2, 0, stride, matrixLayout);

    let mat3 = CoopMatFloatType(mat1);
    let mat4 = CoopMatFloatType(mat2);

    let mat5 = CoopMatUintType(mat1);
    let mat6 = CoopMatUintType(mat2);

    let result = mat1 % mat2;
    let result2 = CoopMatIntType(mat3 % mat4);
    let result3 = CoopMatIntType(mat5 % mat6);

    result.store(outputBuffer, 0, stride, matrixLayout);
    result2.store(outputBuffer, 16, stride, matrixLayout);
    result3.store(outputBuffer, 32, stride, matrixLayout);
}
