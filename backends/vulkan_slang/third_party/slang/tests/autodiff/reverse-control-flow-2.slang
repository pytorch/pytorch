//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=CHK):-target glsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
bool doWork(float x, out float y)
{
    bool retVal = false;
    y = 0;
    for (;;)
    {
        if (x == 0.0)
            break;

        bool exited = (x == 1.0);

        y += x;

        if (!exited)
        {
            if (x < 1.0)
            {
                float b = x * 2.0f;
                y += b;
                exited = true;
            }
        }
        retVal = true;
        break;
    }
    return retVal;
}

[BackwardDifferentiable]
bool doWork2(float x, out float y)
{
    y = 0;

    if (x == 0.0) return false;

    [ForceUnroll]
    for (int i = 0; i < 2; ++i)
    {
        if (x > 0.0)
        {
            y += x;

            if (x == 1.0) break;

            y += x;
        }
        else
        {
            y += x;
        }
    }
    return true;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        var dpx = diffPair(0.5f, 1.0f);
        __bwd_diff(doWork)(dpx, 1.0f);
        outputBuffer[0] = dpx.d;
    }
    {
        var dpx = diffPair(0.5f, 0.0f);
        __bwd_diff(doWork2)(dpx, 1.0);
        outputBuffer[1] = dpx.d;
    }
}

//CHK: (0): note: no checkpoint contexts to report