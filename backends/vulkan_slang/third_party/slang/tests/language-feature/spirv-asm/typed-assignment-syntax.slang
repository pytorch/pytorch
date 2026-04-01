//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK:      2
// CHECK-NEXT: 4
// CHECK-NEXT: 6
// CHECK-NEXT: 8

//
// This test tests a basic application of the spirv_asm block's typed assignment syntax
//
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;
    int n = outputBuffer[i];
    int r = spirv_asm
    {
        // Check that this still works even with no operands on the RHS
        %unused : $$uint = OpUndef;
        // More normal usage
        %two   : $$int = OpConstant 2;
        result : $$int = OpIMul $n %two;
    };
    outputBuffer[i] = r;
}
