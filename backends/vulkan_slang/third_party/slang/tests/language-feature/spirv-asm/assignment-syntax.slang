//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8

//
// This test tests a basic application of the spirv_asm block's assignment syntax
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        %two   = OpConstant $$int 2;
        result = OpIMul $$int $n %two;
    };
    outputBuffer[i] = r;
}
