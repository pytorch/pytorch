//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float3> dpfloat3;

[BackwardDifferentiable]
float diffDot(float3 x, float3 y)
{
    return dot(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat3 dpx = dpfloat3(float3(-0.5, 0.7, 0.8), float3(0.0, 0.0, 0.0));
        dpfloat3 dpy = dpfloat3(float3(0.0, 0.0, 1.0), float3(0.0, 0.0, 0.0));
        __bwd_diff(diffDot)(dpx, dpy, 1.0);
        outputBuffer[0] = dpx.d[0]; // Expect: 0.000000
        outputBuffer[1] = dpx.d[1]; // Expect: 0.000000
        outputBuffer[2] = dpx.d[2]; // Expect: 1.000000
        outputBuffer[3] = dpy.d[0]; // Expect: -0.500000
        outputBuffer[4] = dpy.d[1]; // Expect: 0.700000
        outputBuffer[5] = dpy.d[2]; // Expect: 0.800000
    }
}
