//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
 
float f(float x)
{
    return x*x;
}

[ForwardDerivativeOf(f)]
DifferentialPair<float> df(DifferentialPair<float> x)
{
    var primal = x.p * x.p;
    var diff = 2 * x.p * x.d;
    return DifferentialPair<float>(primal, diff);
}

[ForwardDifferentiable]
float g(float x)
{
    return f(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(3.0, 1.0);

        outputBuffer[0] = f(dpa.p);        // Expect: 9.0
        outputBuffer[1] = __fwd_diff(g)(dpa).d; // Expect: 6.0
    }
}
