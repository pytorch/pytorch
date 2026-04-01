//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
float f(float x)
{
    return sin(x) + x * x;
}

[BackwardDifferentiable]
float outerF(float x)
{
    return f(pow(x, 3));
}

[BackwardDifferentiable]
float df(float x)
{
    return __fwd_diff(outerF)(DifferentialPair<float>(x, 1.0)).d; // x^3 + x^6
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Given f(x) = sin(x^3) + x^6
    // f'(x) = cos(x^3)*(3*x^2) + 6*x^5
    // f''(x) = 6*cos(x^3)*x - 9*sin(x^3)*(x^4) + 30*x^4
    //

    // Expect f''(4) = -0.291
    {
        var p = diffPair(1.0, 0.0);
        __bwd_diff(df)(p, 1.0);
        outputBuffer[0] = p.d;
    }

    // Expect f''(4) = -0.653643
    {
        var p = diffPair(2.0, 0.0);
        __bwd_diff(df)(p, 0.5);
        outputBuffer[1] = p.d;
    }
}
