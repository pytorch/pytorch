//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_simple_single_iter_loop(float y)
{
    float x = 0;

    for (int i = 0; i < 1; i++)
    {
        if (y > 0.6)
        {
            x = y * 10.0f;
            break;
        }
        else
        {
            x = y * 6.0f;
            break;
        }
    }
    
    return x;
}

[BackwardDifferentiable]
float test_nested_if_else_single_iter_loop(float y)
{
    float x = 0;

    for (int i = 0; i < 1; i++)
    {
        if (y > 0.6)
        {
            if (y > 0.8)
            {
                x = y * 10.0f;
                break;
            }
            else
            {
                x = y * 4.0f;
                break;
            }
        }

        x = y * 6.0f;
        break;
    }

    return x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_single_iter_loop)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 10.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);
        
        __bwd_diff(test_simple_single_iter_loop)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 6.0
    }

    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_nested_if_else_single_iter_loop)(dpa, 1.0f);
        outputBuffer[2] = dpa.d; // Expect: 10.0
    }

    {
        dpfloat dpa = dpfloat(0.7, 0.0);

        __bwd_diff(test_nested_if_else_single_iter_loop)(dpa, 1.0f);
        outputBuffer[3] = dpa.d; // Expect: 4.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_nested_if_else_single_iter_loop)(dpa, 1.0f);
        outputBuffer[4] = dpa.d; // Expect: 6.0
    }
}
