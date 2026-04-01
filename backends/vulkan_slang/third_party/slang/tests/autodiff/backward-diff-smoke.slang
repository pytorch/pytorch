//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
float test(float x, float y)
{
    return 2.0f * x + y * 4.0f - x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    dpfloat dpa = dpfloat(2.0, 0.0);
    dpfloat dpb = dpfloat(1.5, 0.0);

    __bwd_diff(test)(dpa, dpb, 1.0f);
    outputBuffer[0] = dpa.d; // Expect: 1
    outputBuffer[1] = dpb.d; // Expect: 4
}
