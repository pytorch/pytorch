//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float f(float x)
{
    return x;
}

[BackwardDifferentiable]
float h(float x, float y)
{
    float m = x + y;
    float n = x - y;
    return m * n + 2 * x * y;
}

[BackwardDifferentiable]
float j(float x, float y)
{
    float m = x / y;
    return m * y;
}

[BackwardDifferentiable]
float k(float x, float y)
{
    float m = -x;
    return m * y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);

        __bwd_diff(f)(dpa, 1.0);

        outputBuffer[0] = dpa.d; // Expect: 1
    }

    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(1.5, 1.0);

        __bwd_diff(h)(dpa, dpb, 1.0);

        outputBuffer[1] = dpa.d; // Expect: (2 * 2.0) + (2 * 1.5) = 7.0
        outputBuffer[2] = dpb.d; // Expect: -(2 * 1.5) + (2 * 2.0) = 1.0
    }

    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(1.5, 1.0);

        __bwd_diff(j)(dpa, dpb, 1.0);

        outputBuffer[3] = dpa.d; // Expect: 1
        outputBuffer[4] = dpb.d; // Expect: 0
    }

    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(1.5, 1.0);

        __bwd_diff(k)(dpa, dpb, 1.0);

        outputBuffer[5] = dpa.d; // Expect: -1.5
        outputBuffer[6] = dpb.d; // Expect: -2.0
    }
}
