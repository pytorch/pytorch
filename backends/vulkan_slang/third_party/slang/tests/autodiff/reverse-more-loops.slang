//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer

RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
float bsdf()
{
    return 0.5;
}

[ForwardDerivativeOf(bsdf)]
DifferentialPair<float> d_bsdf()
{
    return diffPair(0.5f, 1.0f);
}

[BackwardDerivativeOf(bsdf)]
void d_bsdf(float dOut)
{
    outputBuffer[0] += dOut;
}

[BackwardDifferentiable]
float tracePath()
{
    float thp = 1.0;
    float L = 0.0;

    uint depth = 0;

    for (int i = 0; i < 3; ++i)
    {
        if (depth <= 2)
        {
            thp = thp * bsdf();

            L = thp * 1.0;

            if (depth >= 2) break;

            depth = depth + 1;
        }
        else
        {
            L = 0.0;
        }
    }
    
    return L;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        __bwd_diff(tracePath)(1.0); // Expect: 1.0 in outputBuffer[0]
    }
}
