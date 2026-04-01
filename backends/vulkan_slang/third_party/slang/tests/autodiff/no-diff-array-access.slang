//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef DifferentialPair<float3> dpfloat3;

[ForwardDifferentiable]
float f(float x, float[3] y)
{
    return x * no_diff(outputBuffer[4]) + y[2] * x * no_diff(y[1]);
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var dpx = fwd_diff(f)(DifferentialPair<float>(1.0f, 1.0f), DifferentialPair<float[3]>( { 1.0f, 2.0f, 3.0f }, { 1.0f, 2.0f, 3.0f }));
    outputBuffer[0] = dpx.p; // Expect: 6.0
    outputBuffer[1] = dpx.d; // Expect: 12.0
}
