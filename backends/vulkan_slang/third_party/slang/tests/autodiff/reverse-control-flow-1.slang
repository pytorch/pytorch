//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=CHK):-target glsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test_single_branch(float y)
{
    float o;
    if (y > 0.5)
    {
        o = y * 2.0f;
    }
    else
    {
        o = y + 1.0f;
    }

    return o;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_single_branch)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 2.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);
        
        __bwd_diff(test_single_branch)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 1.0
    }
}

//CHK: (0): note: no checkpoint contexts to report