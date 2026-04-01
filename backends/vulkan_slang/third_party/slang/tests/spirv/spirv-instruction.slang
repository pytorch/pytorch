// spirv-instruction.slang
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute 

// Test using a spirv op. 128 is SpvOpIAdd

[[vk::spirv_instruction(128)]]
uint add(uint a, uint b); 

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

[numthreads(4,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadId = dispatchThreadID.x;
    resultBuffer[threadId] = add(threadId, threadId);
}
