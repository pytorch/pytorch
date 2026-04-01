//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float f(float x, float y)
{
    return x > 0.0 ? sqrt(x)*sqrt(x) : y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(0.5, 1.0);

        __bwd_diff(f)(dpa, dpb, 1.0);

        outputBuffer[0] = dpa.d;    // Expect: 1.0
        outputBuffer[1] = dpb.d;    // Expect: 0.0
    }

    {
        dpfloat dpa = dpfloat(-0.3, 1.0);
        dpfloat dpb = dpfloat(0.3, 1.0);

        __bwd_diff(f)(dpa, dpb, 1.0);

        outputBuffer[2] = dpa.d; // Expect: 0.0
        outputBuffer[3] = dpb.d; // Expect: 1.0
    }
}
