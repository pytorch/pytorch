//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -compile-arg -O3 -compute-dispatch 3,1,1 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -compute-dispatch 3,1,1 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -compute-dispatch 3,1,1 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -compute-dispatch 3,1,1 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-mtl -compute -compute-dispatch 3,1,1 -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0  0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadId)
{
    outputBuffer[dispatchThreadID.x] = int((dispatchThreadID.x << 8) | (groupID.x << 4) | (groupThreadID.x));
}
