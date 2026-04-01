//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features half 

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    // We want to test 0
    float value = 0.0f;
    // Produces some somewhat interesting numbers
    if (idx != 0)
    {
        value = (3 << idx);
        
        if ((idx & 1) != 0)
        {
            value = -value;
        }
        
        // Do the recip
        if ((idx & 4) != 0)
        {
            value = 1.0f / value;
        }
    }

    uint r = f32tof16(value);
    
    outputBuffer[idx] = r;
}
