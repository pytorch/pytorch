//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

// inf, -inf, nan, finite
//TEST_INPUT:ubuffer(data=[ 0x7f800000 0xff800000 0x7fffffff 1 ], stride=4):name inputBuffer
RWStructuredBuffer<float> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    float v = inputBuffer[idx];
    
    int flags = 0;
    
    flags |= isnan(v) ? 1 : 0;
    flags |= isfinite(v) ? 2 : 0;
    flags |= isinf(v) ? 4 : 0;
    
    outputBuffer[idx] = flags;
}