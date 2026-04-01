//TEST_CATEGORY(wave, compute)
//DISABLE_TEST:COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST:COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -shaderobj
//TEST(vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0 -shaderobj
//TEST:COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj
//TEST:COMPARE_COMPUTE_EX:-metal -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    value |= WaveActiveAllTrue(idx < 4 ) ? 1 : 0;
    value |= WaveActiveAnyTrue(idx == 2) ? 2 : 0;
    value |= WaveActiveAnyTrue(idx == -1) ? 4 : 0;
    value |= WaveActiveAllTrue(idx == 3) ? 8 : 0;
        
    int sum = WaveActiveSum(idx);
    value |= (sum << 4);
    
    // TODO(JS): 
    // This result is unexpected. I expect 1 * 2 * 1 * 2 = 4. But we get 0 on DX (so disable for now). On CUDA I get 4.
    // int product = WaveActiveProduct((idx & 1) + 1);
    /// value |= (product << 8);    
        
    // TODO(JS): NOTE! This only works with uint, *NOT* int on HLSL/DXC. 
    // We need to update the core module to reflect this.
    uint xor = WaveActiveBitXor(uint(idx + 1));
    value |= int(xor << 12);
    
    outputBuffer[idx] = value;
}
