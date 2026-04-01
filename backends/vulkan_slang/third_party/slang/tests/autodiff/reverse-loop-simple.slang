//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//DISABLE_TEST:SIMPLE(filecheck=CHK):-target glsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

//CHK: note: checkpointing context of 24 bytes associated with function: 'test_simple_loop'
[Differentiable]
float test_simple_loop(float y)
{
    //CHK: note: 20 bytes (FixedArray<float, 5> ) used to checkpoint the following item:
    float t = y;

    //CHK: note: 4 bytes (int32_t) used for a loop counter here:
    for (int i = 0; i < 3; i++)
    {
        t = t * t;
    }

    return t;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_loop)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 8.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_simple_loop)(dpa, 1.0f);
        outputBuffer[1] = dpa.d; // Expect: 0.0131072
    }
}

//CHK-NOT: note