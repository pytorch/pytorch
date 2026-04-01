//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float3> dpfloat3;
typedef float3.Differential dfloat3;

typedef DifferentialPair<float2> dpfloat2;
typedef float2.Differential dfloat2;

[BackwardDifferentiable]
float2 test_reshape(float3 x, float3 y, int i, int j)
{
    float2x3 m = float2x3(x, y);
    let mSmall = float2x2(m);
    return mSmall[i] + mSmall[j];
}

[BackwardDifferentiable]
float3 test_vectorFromScalar(float x)
{
    return float3(x);
}

[BackwardDifferentiable]
float3x3 test_matrixFromScalar(float x)
{
    return float3x3(x);
}

[BackwardDifferentiable]
float2x2 test_matrixConstruct(float a, float b, float c, float d)
{
    return float2x2(a, b, c, d);
}

[BackwardDifferentiable]
float3 test_makeVector(float x, float2 y)
{
    return float3(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat3 dpx = dpfloat3(float3(2.0, 3.0, 4.0), float3(0.0, 0.0, 0.0));
        dpfloat3 dpy = dpfloat3(float3(1.5, 2.5, 3.5), float3(0.0, 0.0, 0.0));

        __bwd_diff(test_reshape)(dpx, dpy, 0, 1, dfloat2(1.0, 2.0));
        outputBuffer[0] = dpx.d.y; // Expect: 2
        outputBuffer[1] = dpy.d.y; // Expect: 2
    }

    {
        DifferentialPair<float> dpx = diffPair(1.0, 0.0);

        __bwd_diff(test_vectorFromScalar)(dpx, dfloat3(2.0));
        outputBuffer[2] = dpx.d; // Expect: 6.0
    }

    {
        DifferentialPair<float> dpx = diffPair(1.0, 0.0);

        __bwd_diff(test_matrixFromScalar)(dpx, float3x3(1.0));
        outputBuffer[3] = dpx.d; // Expect: 9.0
    }
    {
        DifferentialPair<float> dpa = diffPair(1.0, 0.0);
        DifferentialPair<float> dpb = diffPair(1.0, 0.0);
        DifferentialPair<float> dpc = diffPair(1.0, 0.0);
        DifferentialPair<float> dpd = diffPair(1.0, 0.0);

        __bwd_diff(test_matrixConstruct)(dpa, dpb, dpc, dpd, float2x2(1.0, 2.0, 3.0, 4.0));
        outputBuffer[4] = dpa.d; // Expect: 1.0
        outputBuffer[5] = dpb.d; // Expect: 2.0
        outputBuffer[6] = dpc.d; // Expect: 3.0
        outputBuffer[7] = dpd.d; // Expect: 4.0
    }
    {
        DifferentialPair<float> dpx = diffPair(1.0, 0.0);
        dpfloat2 dpy = dpfloat2(float2(1.5, 2.5), float2(0.0, 0.0));

        __bwd_diff(test_makeVector)(dpx, dpy, float3(1.0, 1.5, 2.0));
        outputBuffer[8] = dpx.d; // Expect: 1.0
        outputBuffer[9] = dpy.d.x; // Expect: 1.5
        outputBuffer[10] = dpy.d.y; // Expect: 2.0
    }

}
