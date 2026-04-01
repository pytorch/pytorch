//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;

[BackwardDifferentiable]
float diffAbs(float x)
{
    return abs(x);
}

[BackwardDifferentiable]
float3 diffAbs(float3 x)
{
    return abs(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpx = dpfloat(-3.0, 2.0);
        dpfloat res = __fwd_diff(diffAbs)(dpx);
        outputBuffer[0] = res.p;               // Expect: 3.000000
        outputBuffer[1] = res.d;               // Expect: -2.000000
    }

    {
        dpfloat3 dpx = dpfloat3(float3(2.f, -3.f, -1.f), float3(-2.f, 3.f, -1.f));
        dpfloat3 res = __fwd_diff(diffAbs)(dpx);
        outputBuffer[2] = res.p[0];            // Expect: 2.000000
        outputBuffer[3] = res.d[0];            // Expect: -2.000000
        outputBuffer[4] = res.p[1];            // Expect: 3.000000
        outputBuffer[5] = res.d[1];            // Expect: -3.000000
        outputBuffer[6] = res.p[2];            // Expect: 1.000000
        outputBuffer[7] = res.d[2];            // Expect: 1.000000
    }

    {
        dpfloat dpx = dpfloat(-3.0, 0.0);
        __bwd_diff(diffAbs)(dpx, 2.0);
        outputBuffer[8] = dpx.d; // Expect: -2.000000
    }

    {
        dpfloat3 dpx = dpfloat3(float3(2.f, -3.f, -1.f), float3(0.f, 0.f, 0.f));
        __bwd_diff(diffAbs)(dpx, float3(1.f, 1.f, 1.f));        
        outputBuffer[9] = dpx.d[0];  // Expect: 1.000000
        outputBuffer[10] = dpx.d[1]; // Expect: -1.000000
        outputBuffer[11] = dpx.d[2]; // Expect: -1.000000
    }
}
