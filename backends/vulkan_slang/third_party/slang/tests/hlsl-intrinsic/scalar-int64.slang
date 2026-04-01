//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
// No support for int64_t on D3D11 (no sm 6.0)
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
// No support with Dx12 with dxbc. Needs SM6.0 + dxil
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -profile cs_6_0 -dx12 -use-dxil -shaderobj -render-feature hardware-device
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -render-feature int64
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int64_t idx = int64_t(dispatchThreadID.x);
    
    int64_t ti =0;
    
    ti += max(2, idx);
    ti += min(idx, 1);
    ti += abs(idx - 2);
    ti += clamp(idx * 10, 11, 23);

    // Math
    ti = ti - idx;
    ti = ti + (ti % (idx + 5));
    ti = ti + (ti / (idx + 20));
    
    // Logical ops
    ti = ti | idx;
    ti = ti ^ (idx & (idx + 1));
    
    // Shift
    ti = ti + (idx >> 1);
    ti = ti + (idx << 2);
  
    int64_t v = (ti * 0x400010035435435ll) / 3ll + 7ll - 9ll; 
    
    outputBuffer[uint(idx)] = int(v) ^ int(((v >> 32) & 0xffffffff)); 
}