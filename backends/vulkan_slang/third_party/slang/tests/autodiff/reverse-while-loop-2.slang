//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

struct P
{
    bool terminated;
    bool isTerminated() { return terminated; }
    bool isHit() { return !terminated; }
}

[BackwardDifferentiable]
float test_simple_while(float y)
{
    float t = y;
    P p;
    p.terminated = false;
    int i = 2;

    [MaxIters(3)]
    while (!p.isTerminated())
    {
        i++;
        if (p.isHit())
        {
            if (p.isTerminated()) break;
            t = t * t;
            p.terminated = (i < 5);
        }
        else
        {
            t = t + 1.0;
        }
    }

    return t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_while)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 8.0
    }
}
