//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;

[ForwardDifferentiable]
float f(float x, no_diff float y)
{
    return x * x + y * y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        let rs = __fwd_diff(f)(dpfloat(1.5, 1.0), 2.0);
        outputBuffer[0] = rs.p; // Expect: 6.25
        outputBuffer[1] = rs.d; // Expect: 3.0
    }
}
