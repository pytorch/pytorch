//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8

//
// This test tests that the implicit `OpName` inserted for spirv_asm IDs can
// coexist with an explicit OpName instruction
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        OpConstant $$int %two 2;
        OpName %two "TWO";
        OpIMul $$int result $n %two
    };
    outputBuffer[i] = r;
}
