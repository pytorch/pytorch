//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_simple_nested_loop(float y)
{
    float t = y;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            t = t * (i + j + 1);
        }
    }
    
    return t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_nested_loop)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 12.0 * 1
    }

    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_nested_loop)(dpa, 0.4f);
        outputBuffer[1] = dpa.d; // Expect: 12 * 0.4 = 4.8
    }
}
