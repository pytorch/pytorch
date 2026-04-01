// direct-spirv-control-flow.slang

//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute

// Test direct SPIRV emit on control fl.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

uint test(uint p)
{
    int result = 0;
    if (p == 0)
    {
        result = 5;
    }
    else
    {
        result = 6;
    }
    return result;
}

[numthreads(4,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadId = dispatchThreadID.x;
    uint result = test(threadId);
    resultBuffer[threadId] = result;
}
