//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -compile-arg -O3 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

void writeArray(inout float3 a[4])
{
    a[0] = float3(1, 1, 1);
    a[1] = float3(1, 1, 1);
    a[2] = float3(1, 1, 1);
    a[3] = float3(1, 1, 1);    
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    float3 b[4];
    writeArray(b);
	outputBuffer[dispatchThreadID.x] = int(b[0].x);
}
