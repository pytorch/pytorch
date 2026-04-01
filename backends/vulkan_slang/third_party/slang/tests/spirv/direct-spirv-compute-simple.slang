// direct-spirv-compute-simple.slang
//TESTD:SIMPLE:-target spirv -entry computeMain -stage compute
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute

// Test runinng a shader generated from direct SPIR-V emit.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

[numthreads(4,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadId = dispatchThreadID.x;
    uint result = threadId + 1;
    result = result - 1;
    result = result * 2;
    result = result / 2;
    result = result % 3;
    result = (result ^ 7);
    result = (result & 7);
    result = (result | 8);
    resultBuffer[threadId] = result;
}
