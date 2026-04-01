//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature double
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Double and other unsupported scalar types
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

// inf, -inf, nan, finite
//TEST_INPUT:ubuffer(data=[ 0 0x7ff00000 0 0xfff00000 0xffffffff 0x7fffffff 1 0], stride=4):name inputBuffer
RWStructuredBuffer<uint> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    uint low = inputBuffer[idx * 2 + 0];
    uint high = inputBuffer[idx * 2 + 1];

    double v = asdouble(low, high);
    
    int flags = 0;
    
    flags |= isnan(v) ? 1 : 0;
    flags |= isfinite(v) ? 2 : 0;
    flags |= isinf(v) ? 4 : 0;
    
    outputBuffer[idx] = flags;
}
