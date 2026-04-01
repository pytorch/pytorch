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

[ForwardDifferentiable]
float df(float x)
{
    return __fwd_diff(f)(DifferentialPair<float>(x, 1.0)).d;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Given f(x) = x^4,
    // f''(x) = 12 * x^2
    // Expect f''(4) = 192
    outputBuffer[0] = __fwd_diff(df)(DifferentialPair<float>(4.0, 1.0)).d;
}
