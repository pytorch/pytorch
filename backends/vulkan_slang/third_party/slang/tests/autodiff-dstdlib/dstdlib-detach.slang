//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;

[BackwardDifferentiable]
float diffDetach(float x)
{
    return detach(x);
}

[BackwardDifferentiable]
float2 diffDetach(float2 x)
{
    return detach(x);
}

[BackwardDifferentiable]
float3 diffDetach(float3 x)
{
    return detach(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpx = dpfloat(-5.0, 1.0);
        dpfloat res = __fwd_diff(diffDetach)(dpx);
        outputBuffer[0] = res.p;               // Expect: -5.000000
        outputBuffer[1] = res.d;               // Expect: 0.000000
    }

    {
        dpfloat3 dpx = dpfloat3(float3(2.f, -3.f, -1.f), float3(-2.f, 3.f, -1.f));
        dpfloat3 res = __fwd_diff(diffDetach)(dpx);
        outputBuffer[2] = res.p[0];            // Expect: 2.000000
        outputBuffer[3] = res.d[0];            // Expect: 0.000000
        outputBuffer[4] = res.p[1];            // Expect: -3.000000
        outputBuffer[5] = res.d[1];            // Expect: 0.000000
        outputBuffer[6] = res.p[2];            // Expect: -1.000000
        outputBuffer[7] = res.d[2];            // Expect: 0.000000
    }

    {
        dpfloat dpx = dpfloat(-5.0, 1.0);
        __bwd_diff(diffDetach)(dpx, 1.0);
        outputBuffer[8] = dpx.d; // Expect: 0.000000
    }

    {
        dpfloat2 dpx = dpfloat2(float2(1.0, -2.0), float2(1.0, 1.0));
        __bwd_diff(diffDetach)(dpx, float2(2.0, -3.0));
        outputBuffer[9] = dpx.d[0]; // Expect: 0.000000
        outputBuffer[10] = dpx.d[1]; // Expect: 0.000000
    }
}
