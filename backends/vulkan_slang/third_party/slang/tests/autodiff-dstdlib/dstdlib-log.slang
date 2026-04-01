//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float2> dpfloat2;

[BackwardDifferentiable]
float diffLog(float x)
{
    return log(x);
}

[BackwardDifferentiable]
float2 diffLog(float2 x)
{
    return log(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(10.0, 0.0);
        __bwd_diff(diffLog)(dpa, 3.0);
        outputBuffer[0] = dpa.d; // Expect: 0.300000
    }

    {
        dpfloat2 dpx = dpfloat2(float2(2.0, 5.0), float2(0.0, 0.0));
        __bwd_diff(diffLog)(dpx, float2(1.0, 1.0));
        outputBuffer[1] = dpx.d[0]; // Expect: 0.500000
        outputBuffer[2] = dpx.d[1]; // Expect: 0.200000
    }
}
