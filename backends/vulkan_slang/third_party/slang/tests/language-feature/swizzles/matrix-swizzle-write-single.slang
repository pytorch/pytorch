//TEST(compute):COMPARE_COMPUTE: -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE: -vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl -output-using-type

// Test that writes to single matrix elements with swizzles work

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint tid : SV_GroupIndex)
{
    // 0 0
    // 0 0
    float2x2 m = float2x2(0);

    // 0 2
    // 0 0
    const float x = m._m01 = 2;

    outputBuffer[tid] = x + m[tid/2][tid%2];
}
