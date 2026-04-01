//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_if_only(float y)
{
    float o = y + 1.0f;
    if (y > 0.5)
    {
        o = y * 2.0f;
    }

    return o;
}

[BackwardDifferentiable]
float test_nested_ifelse(float y)
{
    float o = 0.f;
    if (y > 0.5)
    {
        o = y * 2.0f;
    }
    else
    {
        if (y > 0.25)
        {
            o = y * 3.0f + 2.0f;
        }
        else
        {
            o = y * 4.0f;
        }
    }

    return o;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_nested_ifelse)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 2.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);
        
        __bwd_diff(test_nested_ifelse)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 3.0
    }

    {
        dpfloat dpa = dpfloat(0.2, 0.0);
        
        __bwd_diff(test_nested_ifelse)(dpa, 1.0f);
        outputBuffer[2] = dpa.d; // Expect: 4.0
    }

    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_if_only)(dpa, 1.0f);
        outputBuffer[3] = dpa.d; // Expect: 2.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);
        
        __bwd_diff(test_if_only)(dpa, 1.0f);
        outputBuffer[4] = dpa.d; // Expect: 1.0
    }
}
