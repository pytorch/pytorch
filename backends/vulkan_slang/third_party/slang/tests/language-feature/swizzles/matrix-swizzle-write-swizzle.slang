//TEST(compute):COMPARE_COMPUTE: -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE: -vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl -output-using-type

// Test that writing to swizzles of matrix swizzles works correctly

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint tid : SV_GroupIndex)
{
    float2x2 m = float2x2(0);

    // 1 4
    // 3 2
    m._m00_m01_m10_m11.xwzy.wzyx.wzyx = float4(1,2,3,4);

    outputBuffer[tid] = m[tid/2][tid%2];
}
