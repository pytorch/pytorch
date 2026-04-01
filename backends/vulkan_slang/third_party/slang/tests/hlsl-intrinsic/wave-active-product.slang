//TEST_CATEGORY(wave, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj -render-feature hardware-device
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature hardware-device
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0 -shaderobj
//TEST:COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-metal -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = int(dispatchThreadID.x);
       
#if 1
    if (idx < 3)
    {
        // Diverge!!
        outputBuffer[idx] = -1;
        return;
    }
    outputBuffer[idx] = WaveActiveProduct(idx);
#else

    /// NOTE! Can't say I totally understand WaveActiveProduct. 
    /// The following returns 0x240 on CUDA - which is what I'd expect
    /// On DX12, it returns 0

    outputBuffer[idx] = WaveActiveProduct((idx & 3) + 1);
#endif
}
