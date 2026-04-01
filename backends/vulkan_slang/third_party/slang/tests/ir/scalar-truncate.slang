//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
     {
        int16_t a = 30000;      // 0x00007530 -> 0x7530 (30000) in 16-bit.
        int16_t b = int16_t(-200000);    // 0xfffcf2c0 -> Truncated to 0xf2c0 (-3392) in 16-bit.
        outputBuffer[0] = asuint((int)a);
        outputBuffer[1] = asuint((int)b);
    }
}
