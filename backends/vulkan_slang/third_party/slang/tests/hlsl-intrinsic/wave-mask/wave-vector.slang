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
    const WaveMask mask = 0xff;
    
    const int idx = int(dispatchThreadID.x);
       
    int2 v0 = int2(idx + 1, idx + 2);   
    float2 v1 = float2(idx + 2, idx + 3);
    // NOTE! dxc only supports bit ops on uint and associated types NOT int
    uint2 uv0 = v0;
    
    int2 r0 = WaveMaskSum(mask, v0);
    float2 r1 = WaveMaskSum(mask, v1);
    int2 r2 = int2(WaveMaskBitXor(mask, uv0));
    int2 r3 = int2(WaveMaskBitOr(mask, uv0));
    int2 r4 = int2(WaveMaskBitAnd(mask, uv0));
    
    int2 r = r0 + int2(r1) + r2 + r3 + r4;
   
    outputBuffer[idx] = r.x + r.y;
}