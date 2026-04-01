//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_multi_return(float y)
{
    if (y > 0.6)
    {
        if (y > 0.8)
        {
            return y * 10.0f;
        }
        else
        {
            return y * 4.0f;
        }
    }
    return y * 6.0f;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_multi_return)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 10.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);
        
        __bwd_diff(test_multi_return)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 6.0
    }

    {
        dpfloat dpa = dpfloat(0.7, 0.0);
        
        __bwd_diff(test_multi_return)(dpa, 1.0f);
        outputBuffer[2] = dpa.d; // Expect: 4.0
    }
}
