//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_simple_while(float y)
{
    float t = y;

    bool keepGoing = true;
    int i = 2;

    [MaxIters(3)]
    do 
    {
        i++;
        t = t * t;

        keepGoing = (i < 5);
    } while (keepGoing);

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

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_simple_while)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 0.0131072
    }
}
