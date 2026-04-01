//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type -render-features half
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type -render-features half
// Not supported in WGSL: Double and other unsupported scalar types
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<double> dpdouble;
typedef DifferentialPair<half> dphalf;
typedef float.Differential dfloat;
typedef double.Differential ddouble;
typedef half.Differential dhalf;

[BackwardDifferentiable]
float f(double x)
{
    return (float)x;
}

[BackwardDifferentiable]
float h(float x, float y)
{
    float m = x + y;
    float n = x - y;
    return m * n + 2 * x * y;
}

[BackwardDifferentiable]
float j(half x, half y)
{
    float m = x / y;
    return m * y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpdouble dpa = dpdouble(2.0, 1.0);

        __bwd_diff(f)(dpa, 1.0);

        outputBuffer[0] = (float)dpa.d; // Expect: 1
    }

    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(1.5, 1.0);

        __bwd_diff(h)(dpa, dpb, 1.0);

        outputBuffer[1] = dpa.d; // Expect: (2 * 2.0) + (2 * 1.5) = 7.0
        outputBuffer[2] = dpb.d; // Expect: -(2 * 1.5) + (2 * 2.0) = 1.0
    }

    {
        dphalf dpa = dphalf((half)2.0, (half)1.0);
        dphalf dpb = dphalf((half)1.5, (half)1.0);

        __bwd_diff(j)(dpa, dpb, 1.0);

        outputBuffer[3] = dpa.d; // Expect: 1
        outputBuffer[4] = dpb.d; // Expect: 0
    }

}
