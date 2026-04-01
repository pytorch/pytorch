//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float2x2> dpmat2;

[BackwardDifferentiable]
float2x2 diffMul(float2x2 a, float2x2 b)
{
    return mul(a, b);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    dpmat2 dpa = dpmat2(float2x2(1.0, 2.0, 3.0, 4.0), float2x2(0.0, 0.0, 0.0, 0.0));
    dpmat2 dpb = dpmat2(float2x2(5.0, 6.0, 7.0, 8.0), float2x2(0.0, 0.0, 0.0, 0.0));
    float2x2 dOut = float2x2(1.0, -2.0, -3.0, 4.0);
    __bwd_diff(diffMul)(dpa, dpb, dOut);
    outputBuffer[0] = dpa.d[0][0]; // Expect: -7.000000
    outputBuffer[1] = dpa.d[0][1]; // Expect: -9.000000
    outputBuffer[2] = dpa.d[1][0]; // Expect: 9.000000
    outputBuffer[3] = dpa.d[1][1]; // Expect: 11.000000
    outputBuffer[4] = dpb.d[0][0]; // Expect: -8.000000
    outputBuffer[5] = dpb.d[0][1]; // Expect: 10.000000
    outputBuffer[6] = dpb.d[1][0]; // Expect: -10.000000
    outputBuffer[7] = dpb.d[1][1]; // Expect: 12.000000
}
