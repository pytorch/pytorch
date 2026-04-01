//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    int3 a = { idx + 10, idx - 9, idx + 3};
    int3 b = { idx * 2, idx * 3, idx * 10};
    
    int3 t = { 0, 0, 0};
    
    t += max(a, b);
    t += min(a, b);
    t += abs(a);
    t += b % 5;
    
    t += clamp(a, int3(10), int3(23));
   
    // Swizzle
    t += a.zyx;
    // Swizzle from scalar
    t += idx.xxx;
       
    outputBuffer[idx] = t.x + t.y + t.z;
}