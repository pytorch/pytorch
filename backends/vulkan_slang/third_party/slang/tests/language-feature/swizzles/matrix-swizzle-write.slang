//TEST(compute):COMPARE_COMPUTE: -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE: -vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl -output-using-type

// Test that matrix swizzle writes work correctly
// Matrix swizzles can either be one or zero indexed
// Reference: https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint tid : SV_GroupIndex)
{
    // 0 0
    // 0 0
    float2x2 m = float2x2(0);

    // 0 1
    // 4 0
    m._12_21 = float2(1, 4);

    // 2 1
    // 4 3
    m._m00_m11 = float2(2, 3);

    outputBuffer[tid] = m[tid/2][tid%2];
}
