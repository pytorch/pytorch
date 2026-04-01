//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;

[BackwardDifferentiable]
float diffSqrt(float x)
{
    return sqrt(x);
}

[BackwardDifferentiable]
float2 diffSqrt(float2 x)
{
    return sqrt(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpx = dpfloat(2.0, 1.0);
        dpfloat res = __fwd_diff(diffSqrt)(dpx);
        outputBuffer[0] = res.p;               // Expect: 1.414214
        outputBuffer[1] = res.d;               // Expect: 0.353553
    }

    {
        dpfloat dpx = dpfloat(100.0, -3.0);
        dpfloat res = __fwd_diff(diffSqrt)(dpx);
        outputBuffer[2] = res.p;               // Expect: 10.000000
        outputBuffer[3] = res.d;               // Expect: -0.150000
    }

    {
        dpfloat dpx = dpfloat(0.0, 1.0);
        dpfloat res = __fwd_diff(diffSqrt)(dpx);
        outputBuffer[4] = res.p; // Expect: 0.000000
        outputBuffer[5] = res.d; // Expect: 1581.138916
    }

    {
        dpfloat2 dpx = dpfloat2(float2(10.0, 3.0), float2(0.0, 0.0));
        __bwd_diff(diffSqrt)(dpx, float2(1.0, 2.0));
        outputBuffer[6] = dpx.d[0]; // Expect: 0.158114
        outputBuffer[7] = dpx.d[1]; // Expect: 0.577350
    }

    {
        var dpx = diffPair(float2x2(4.0, 9.0, 16.0, 25.0), float2x2(0.0, 0.0, 0.0, 0.0));
        __bwd_diff(sqrt)(dpx, float2x2(1.0, 2.0, 3.0, 4.0));
        outputBuffer[8] = dpx.d[0][0]; // Expect: 0.25
        outputBuffer[9] = dpx.d[0][1]; // Expect: 0.3333
        outputBuffer[10] = dpx.d[1][0]; // Expect: 0.375
        outputBuffer[11] = dpx.d[1][1]; // Expect: 0.4
    }
}
