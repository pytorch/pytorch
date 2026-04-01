//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_simple_switch(float y, int i)
{
    float o;
    switch (i)
    {
        case 0:
        case 1:
            o = y * 2.0;
            break;
        
        case 2:
            o = y * 3.0;
            break;
        
        default:
            o = y;
    }

    return o;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_switch)(dpa, 1, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 2.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);
        
        __bwd_diff(test_simple_switch)(dpa, 2, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 3.0
    }

    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_switch)(dpa, 3, 1.0f);
        outputBuffer[2] = dpa.d; // Expect: 1.0
    }
}
