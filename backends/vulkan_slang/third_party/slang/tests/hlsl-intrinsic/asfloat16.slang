// Float16 not supported on CPU currently
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type
// Doesn't work on FXC
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile sm_6_2 -render-feature half -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -render-feature int16,half -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features half -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    // 1
    float16_t m = asfloat16(uint16_t(0x3c00));

    float16_t v = asfloat16(uint16_t(0x3c00 + (idx << 8))) + m;
    float f = v;

    outputBuffer[idx] = f;
}