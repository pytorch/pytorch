//TEST(compute):COMPARE_COMPUTE: -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE: -vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl -output-using-type

// Test that matrix swizzle writes work correctly
// Matrix swizzles can either be one or zero indexed
// Reference: https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float2x2> outputBuffer;

struct MySubscriptable
{
    float2x2 m;
    __subscript() -> float2x2
    {
        get { return m; }
        set { m = newValue; }
    }
};

[numthreads(1, 1, 1)]
void computeMain(uint tid : SV_GroupIndex)
{
    MySubscriptable s;
    s.m = float2x2(0,0,0,0);

    // _ 1
    // 4 5
    s[]._12_21_22 = float3(1, 4, 5);

    // 2 1
    // 4 3
    s[]._m00_m11 = float2(2, 3);

    outputBuffer[tid] = s.m;
}
