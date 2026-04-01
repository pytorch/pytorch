//TEST_CATEGORY(wave-mask, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature hardware-device
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    WaveMask mask = 0xff;
    
    int idx = int(dispatchThreadID.x);
    
    float2 v1 = float2(1, 1 << idx);
    
    int r0 = WaveMaskPrefixSum(mask, 1 << idx);
    float2 r1 = WaveMaskPrefixSum(mask, v1);
    
    int r2 = int(r1.x) + int(r1.y) - idx;
    
    outputBuffer[idx] = r0 + (r2 << 16);
}