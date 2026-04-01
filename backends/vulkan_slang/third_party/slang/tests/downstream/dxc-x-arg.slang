//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -compile-arg -O3 -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

// We are going to pass -Gis to DXC.
// Test can only work on DXC.
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -Xslang... -Xdxc -Gis -X. -shaderobj 

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadID.x] = dispatchThreadID.x;
}