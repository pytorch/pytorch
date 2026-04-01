//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float f(float x)
{
    float k = x > 1e-8 ? atan2(x, x) : 0.0;
    return k;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = diffPair(2.0);

        __bwd_diff(f)(dpa, 1.0);

        outputBuffer[0] = dpa.d; // Expect: 0.0
    }

    {
        dpfloat dpa = diffPair(0.0,);

        __bwd_diff(f)(dpa, 1.0);

        outputBuffer[1] = dpa.d; // Expect: 0.0, but if select is not turned into if-else,
                                 // this will be 'inf' or 'nan'
    }
}
