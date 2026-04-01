// direct-spirv-control-flow-2.slang

//TESTD:SIMPLE:-target spirv -entry computeMain -stage compute
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type

// Test direct SPIR-V emit on control flows.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

uint test(uint p)
{
    int result = 0;
    for (int i = 0; i < 5; i++)
    {
        result += i*2;
    }
    switch (p)
    {
    case 0:
        result = result - 1;
        break;
    case 1:
        result = result + 1;
        break;
    default:
        result = result * 2;
        break;
    }
    if (p > 2)
    {
        switch (p)
        {
        case 3:
            result++;
            break;
        }
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
