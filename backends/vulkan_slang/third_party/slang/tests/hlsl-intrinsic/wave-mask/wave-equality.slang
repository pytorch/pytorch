//TEST_CATEGORY(wave-mask, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0 -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const WaveMask mask = 0xf;
    
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    // Scalar
    
    value |= WaveMaskAllEqual(mask, idx * 0 + 1)  ? 1 : 0;     // true
    value |= WaveMaskAllEqual(mask, idx & 2)      ? 2 : 0;     // false
   
    // Vector
   
    int2 v0 = int2(idx & 0xf0, (idx & 0xf00) + 1);      // (0, 1)
    int2 v1 = int2(idx & 2, (idx & 2) + 1);
    
    value |= WaveMaskAllEqual(mask, v0)           ? 0x10 : 0;     // true
    value |= WaveMaskAllEqual(mask, v1)           ? 0x20 : 0;     // false
    
    outputBuffer[idx] = value;
}