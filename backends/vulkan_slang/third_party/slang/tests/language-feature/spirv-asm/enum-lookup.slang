//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8

//
// This tests that we can look up a type qualified enum in a spirv block
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        // MemorySemanticsAcquire happens to have value 0x2
        %two   : $$int = OpConstant MemorySemanticsAcquire;
        result : $$int = OpIMul $n %two
    };
    outputBuffer[i] = r;
}
