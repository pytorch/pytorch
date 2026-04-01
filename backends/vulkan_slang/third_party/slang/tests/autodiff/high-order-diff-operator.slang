//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[ForwardDifferentiable]
float mySqr(float x)
{
    return x * x;
}

[ForwardDifferentiable]
float f(float x)
{
    return mySqr(x * x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Given f(x) = x^4,
    // f''(x) = 12 * x^2
    // Expect f''(4) = 192
    float.Differential t = 2;
    outputBuffer[0] = __fwd_diff(__fwd_diff(f))(
                          DifferentialPair<DifferentialPair<float>>(
                              DifferentialPair<float>(4.0, 1.0), DifferentialPair<float>(1.0, 0.0))) .d.d;

    // sin''(x) = cos'(x) = -sin(x).
    // Expect sin''(Pi/2) = -1.
    outputBuffer[1] = __fwd_diff(__fwd_diff(sin))(
                          DifferentialPair<DifferentialPair<float>>(
                              DifferentialPair<float>(float.getPi()/2, 1.0), DifferentialPair<float>(1.0, 0.0))).d.d; 
}
