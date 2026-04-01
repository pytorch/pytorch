//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type -emit-spirv-directly -skip-spirv-validation

// CHECK: type: uint32_t
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 1

//TEST_INPUT:ubuffer(data=[1.0 2.0 3.0 4.0], stride=4, count=256),name=input1
ByteAddressBuffer input1;

//TEST_INPUT:ubuffer(data=[1.0 3.0 2.0 4.0], stride=4, count=256),name=input2
ByteAddressBuffer input2;

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

typealias CoopMatType = CoopMat<float, CoopMatScope::Subgroup, 16, 16, CoopMatMatrixUse::MatrixAccumulator>;

[numthreads(32, 1, 1)]
void computeMain(uint3 threadIndex : SV_DispatchThreadID)
{
    let stride = 4;
    let matrixLayout = CoopMatMatrixLayout::RowMajor;

    let mat1 = CoopMatType.load(input1, 0, stride, matrixLayout);
    let mat2 = CoopMatType.load(input2, 0, stride, matrixLayout);

    uint32_t equals = mat1 == mat2 ? 1 : 0;
    uint32_t lessThan = mat1 < mat2 ? 1 : 0;
    uint32_t lessThanOrEquals = mat1 <= mat2 ? 1 : 0;

    outputBuffer[0] = equals;
    outputBuffer[1] = lessThan;
    outputBuffer[2] = lessThanOrEquals;
}
