//TEST_CATEGORY(wave, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature hardware-device
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0 -shaderobj
//TEST:COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-metal -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = int(dispatchThreadID.x);
       
    int2 v0 = int2(idx + 1, idx + 2);   
    float2 v1 = float2(idx + 2, idx + 3);
    // NOTE! dxc only supports bit ops on uint and associated types NOT int
    uint2 uv0 = v0;
    
    int2 r0 = WaveActiveSum(v0);
    float2 r1 = WaveActiveSum(v1);
    int2 r2 = int2(WaveActiveBitXor(uv0));
    int2 r3 = int2(WaveActiveBitOr(uv0));
    int2 r4 = int2(WaveActiveBitAnd(uv0));
    
    int2 r = r0 + int2(r1) + r2 + r3 + r4;
   
    outputBuffer[idx] = r.x + r.y;
}
