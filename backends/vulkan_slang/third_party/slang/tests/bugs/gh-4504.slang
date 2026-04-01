//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-slang -dx12 -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-mtl -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

// note: `WorkgroupSize()` resolves into a compile time constant,
// it is possible `size.x == 5` and others will resolve into `true`
// at compile time. A compute test is required.
[shader("compute")]
[numthreads(5, 2, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID)
{
    // CHECK: 1
    int3 size = WorkgroupSize();
    outputBuffer[0] = true
        && size.x == 5
        && size.y == 2
        && size.z == 1
        ;
}
