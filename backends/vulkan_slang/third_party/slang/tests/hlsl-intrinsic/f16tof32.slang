//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features half

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    uint values[] = { 0, 0xC600, 0x4A00, 0xCE00, 0x2555, 0xA155, 0x1D55, 0x9955 };
    
    outputBuffer[idx] = f16tof32(values[idx]);
}
