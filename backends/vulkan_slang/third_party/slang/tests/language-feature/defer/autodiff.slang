//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[Differentiable]
float testFunc(float a)
{
    float x = a / (abs(a) + 0.5f);
    {
        defer x = sqrt(x);

        x = x * 0.5f + 0.5f;

        if (a < 0)
        {
            x += 1.0f;
            // NOTE suprising but correct behaviour here: 'defer' occurs after
            // the return statement's value has been computed, so mutating 'x'
            // no longer affects anything.
            return x;
        }

        x += 0.5f;
    }
    return x;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = testFunc(0);
    outputBuffer[1] = testFunc(0.5);
    outputBuffer[2] = testFunc(-0.5);

    DifferentialPair<float> d1 = diffPair(0.0);
    bwd_diff(testFunc)(d1, 1.0);
    DifferentialPair<float> d2 = diffPair(0.5);
    bwd_diff(testFunc)(d2, 1.0);
    DifferentialPair<float> d3 = diffPair(-0.5);
    bwd_diff(testFunc)(d3, 1.0);

    outputBuffer[3] = d1.d;
    outputBuffer[4] = d2.d;
    outputBuffer[5] = d3.d;

    d1 = diffPair(0.0, 1.0);
    d2 = diffPair(0.5, 1.0);
    d3 = diffPair(-0.5, 1.0);

    outputBuffer[6] = fwd_diff(testFunc)(d1).d;
    outputBuffer[7] = fwd_diff(testFunc)(d2).d;
    outputBuffer[8] = fwd_diff(testFunc)(d3).d;
}
