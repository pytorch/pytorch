//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;

[BackwardDifferentiable]
float diffPow(float x, float y)
{
    return pow(x, y);
}

[BackwardDifferentiable]
float2 diffPow(float2 x, float2 y)
{
    return pow(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpx = dpfloat(5.0, 1.0);
        dpfloat dpy = dpfloat(3.0, 0.0);
        dpfloat res = __fwd_diff(diffPow)(dpx, dpy);
        outputBuffer[0] = res.p;               // Expect: 125.000000
        outputBuffer[1] = res.d;               // Expect: 75.000000
    }

    {
        dpfloat2 dpx = dpfloat2(float2(10.0, 3.0), float2(0.0, -2.0));
        dpfloat2 dpy = dpfloat2(float2(2.5, 4.0), float2(1.0, 1.0));
        dpfloat2 res = __fwd_diff(diffPow)(dpx, dpy);
        outputBuffer[2] = res.p[0];            // Expect: 316.227722
        outputBuffer[3] = res.d[0];            // Expect: 728.141235
        outputBuffer[4] = res.p[1];            // Expect: 81.000000
        outputBuffer[5] = res.d[1];            // Expect: -127.012398
    }

    {
        dpfloat2 dpx = dpfloat2(float2(2.0, 4.0), float2(0.0, 0.0));
        dpfloat2 dpy = dpfloat2(float2(3.0, -2.0), float2(0.0, 0.0));
        __bwd_diff(diffPow)(dpx, dpy, float2(1.0, 1.0));
        outputBuffer[6] = dpx.d[0]; // Expect: 12.000000
        outputBuffer[7] = dpx.d[1]; // Expect: -0.031250
        outputBuffer[8] = dpy.d[0]; // Expect: 5.545177
        outputBuffer[9] = dpy.d[1]; // Expect: 0.086643
    }
}
