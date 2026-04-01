// TODO(JS): Disabled because fails on windows CI.
// Probably because on CI it is using std output to detect errors/warnings, not looking at the return code. 

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    float v = 0.0f;

    // Test to check handling of + and -inf
    if (idx == 0)
    {
        v = 5e+40;
    }
    else if (idx == 1)
    {
        v = -5e+40;
    }
    
    outputBuffer[idx] = v;
}