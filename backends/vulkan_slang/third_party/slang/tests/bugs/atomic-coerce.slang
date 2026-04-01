//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj 
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Use of traditional atomics intrinsics (InterlockedXXX functions)
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

groupshared uint gs_values[2];

[shader("compute")]
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = (int)dispatchThreadID.x;
    
    // Initialize first
    if (index < 2)
    {
        gs_values[index] = 2;
    }

    GroupMemoryBarrierWithGroupSync();
 
    // NOTE! We have to cast to uint, to make atomic work
    InterlockedAdd(gs_values[index & 1], uint(index * index));
           
    outputBuffer[index] = gs_values[index & 1];
}
